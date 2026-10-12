"""Independent CPU benchmark using existing models; never overwrite prior runs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast import config, tracking
from gridcast.data import dataset, eia, quality
from gridcast.evaluation import backtest, metrics
from gridcast.experiments import zoo
from gridcast.probabilistic.conformal import calibrate


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_data(df: pd.DataFrame) -> dict:
    if df.index.tz is None or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("Data require an ordered unique timezone-aware index")
    expected = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    if not df.index.as_unit("ns").equals(expected.as_unit("ns")) or not np.isfinite(df.demand).all():
        raise ValueError("Incomplete hourly demand grid")
    raw = eia.read_raw()
    raw = raw[(raw.utc_end >= config.DATA_START) & (raw.utc_end < config.DATA_END)]
    rebuilt, report = quality.clean_demand(raw)
    np.testing.assert_allclose(rebuilt.demand, df.demand, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(rebuilt.is_imputed, df.is_imputed)
    local = df.index.tz_convert(config.LOCAL_TZ).tz_localize(None)
    return {"processed_sha256": fingerprint(dataset.PROCESSED),
            "manifest_sha256": fingerprint(dataset.MANIFEST),
            "raw_sha256": fingerprint(config.DATA_RAW / eia.RAW_NAME),
            "raw_clean_reconstruction_matches": True, "quality": report.to_dict(),
            "local_duplicate_hours_dst": int(local.duplicated().sum()),
            "dataset_version": json.loads(dataset.MANIFEST.read_text())["dataset_version"]}


def paired_daily_bootstrap(scored: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Seven-day moving blocks within folds, weighted by number of scored hours."""
    rows = []
    rng = np.random.default_rng(seed)
    for h, ph in scored.groupby("h"):
        base = ph[ph.model == "seasonal_naive_week"].set_index(["fold", "origin"])
        for name, g in ph.groupby("model"):
            merged = g.merge(base[["yhat"]], left_on=["fold", "origin"], right_index=True,
                             suffixes=("", "_base"), validate="one_to_one")
            merged["difference"] = abs(merged.y - merged.yhat) - abs(merged.y - merged.yhat_base)
            merged["date"] = merged.origin.dt.floor("D")
            daily = merged.groupby(["fold", "date"]).difference.agg(["sum", "count"])
            boot_sum, boot_count = np.zeros(500), np.zeros(500)
            for _, d in daily.groupby(level=0):
                a = d.to_numpy()
                starts = rng.integers(0, len(a) - 7 + 1, size=(500, int(np.ceil(len(a) / 7))))
                idx = (starts[..., None] + np.arange(7)).reshape(500, -1)[:, :len(a)]
                samples = a[idx].sum(axis=1)
                boot_sum += samples[:, 0]
                boot_count += samples[:, 1]
            lo, hi = np.quantile(boot_sum / boot_count, [0.025, 0.975])
            rows.append({"model": name, "h": h, "mae_difference_vs_weekly_mw": merged.difference.mean(),
                         "ci95_low": lo, "ci95_high": hi, "n": len(merged)})
    return pd.DataFrame(rows)


def run() -> Path:
    start = time.perf_counter()
    out = config.RESULTS / "forecasting_baselines" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    df = dataset.load()
    data = validate_data(df)
    names = ["persistence", "seasonal_naive_24h", "seasonal_naive_week", "ridge", "lightgbm"]
    registry = zoo.classical()
    predictions, timings = [], {}
    for name in names:
        print(f"Revalidate {name}", flush=True)
        pred, timing = backtest.run_backtest(registry[name], df, config.TEST_FOLDS, config.HEADLINE_HORIZONS,
                                              regimes=["none"])
        pred.to_parquet(out / f"{name}.parquet", index=False)
        predictions.append(pred)
        timings[name] = timing
    pred = pd.concat(predictions, ignore_index=True)
    scored = backtest.scoreable(pred)
    ends = {f.name: f.test_end for f in config.TEST_FOLDS}
    scored = scored[scored.target_time < scored.fold.map(ends)].copy()
    if scored.duplicated(["model", "fold", "origin", "h"]).any():
        raise ValueError("Duplicate forecast keys")
    counts = scored.groupby(["fold", "h", "model"]).size().unstack("model")
    if not counts.eq(counts.iloc[:, 0], axis=0).all().all():
        raise ValueError("Models do not have common evaluation windows")
    scales = {f.name: metrics.naive_scale(df.loc[(df.index >= f.train_start) & (df.index < f.cal_start), "demand"])
              for f in config.TEST_FOLDS}
    scored["scaled_ae"] = abs(scored.y - scored.yhat) / scored.fold.map(scales)
    scored = backtest.add_slices(scored)
    for label, by in {"overall": ["model", "regime", "h"], "by_fold": ["model", "regime", "h", "fold"],
                      "by_season": ["model", "regime", "h", "season"],
                      "by_demand_regime": ["model", "regime", "h", "load_regime"]}.items():
        table = metrics.score_frame(scored, None, by)
        mase = scored.groupby(by, observed=True).scaled_ae.mean().rename("mase").reset_index()
        table.merge(mase, on=by).to_csv(out / f"{label}.csv", index=False)
    paired_daily_bootstrap(scored).to_csv(out / "paired_bootstrap.csv", index=False)
    intervals = calibrate(pred, kind="abs", mode="static")
    intervals = backtest.scoreable(intervals)
    intervals = intervals[intervals.target_time < intervals.fold.map(ends)]
    metrics.score_frame(intervals, None, ["model", "regime", "h"]).to_csv(out / "conformal_static.csv", index=False)
    record = {"git": tracking.git_state(), "device": "cpu", "versions": tracking.versions(),
              "hardware": tracking.hardware(), "data": data, "seed": 0,
              "folds": [tracking.fold_dict(f) for f in config.TEST_FOLDS], "timings": timings,
              "mase_scales_mw": scales, "wall_seconds": time.perf_counter() - start,
              "source_sha256": {str(p.relative_to(config.ROOT)): fingerprint(p)
                                for p in (config.ROOT / "src").rglob("*.py")},
              "lock_sha256": fingerprint(config.ROOT / "uv.lock"),
              "bootstrap": "500 replicates, seven-day blocks within each fold, weighted by valid hours",
              "test_boundary": "origin and target both within each test fold; exclude imputed targets"}
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    print(out, flush=True)
    return out


if __name__ == "__main__":
    run()

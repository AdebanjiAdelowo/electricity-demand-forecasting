"""Pre-test interval selection, frozen before exploratory primary-fold scoring."""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast import config, tracking
from gridcast.data import dataset
from gridcast.evaluation import backtest, metrics
from gridcast.experiments.revalidate import fingerprint
from gridcast.experiments.weather_uncertainty import interval_table, static_intervals
from gridcast.features.calendar import season_of
from gridcast.models.tabular import LightGBMForecaster, RidgeForecaster
from gridcast.probabilistic.online import OnlineConfig, calibrate_online

ALPHAS = (0.2, 0.1, 0.05)
DAYS = (7, 14, 28, 60)
BASE = config.RESULTS / "forecasting_baselines/20261012T093203Z"
DEV = config.Fold(
    "interval_dev",
    pd.Timestamp("2023-01-24", tz="UTC"),
    pd.Timestamp("2023-04-25", tz="UTC"),
    pd.Timestamp("2023-07-01", tz="UTC"),
    pd.Timestamp("2024-04-01", tz="UTC"),
)
PROTOCOL = {
    "fold": tracking.fold_dict(DEV),
    "models": ["ridge", "lightgbm"],
    "horizons": config.HEADLINE_HORIZONS,
    "window_days": DAYS,
    "min_scores": 168,
    "alphas": ALPHAS,
    "feedback_delay_hours": 0,
    "coverage_tolerance": 0.02,
    "selection": "per model/horizon: minimum 90% Winkler among 88-92% coverage candidates; tie shorter window",
    "failure": "retain static if no eligible candidate; never relax tolerance",
    "complete_keys_required": True,
    "budget_seconds": 1200,
    "budget_peak_bytes": 1073741824,
    "bootstrap": {"seed": 0, "replicates": 500, "block_days": [7, 14, 28]},
    "test_status": "previously inspected exploratory folds, not independent confirmation",
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


ACTIVE_MEMORY_LIMIT = PROTOCOL["budget_peak_bytes"]


def guard():
    # macOS ru_maxrss is bytes. This repository's execution host is macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak > ACTIVE_MEMORY_LIMIT:
        raise RuntimeError(f"Peak RSS {peak} exceeds {ACTIVE_MEMORY_LIMIT} byte budget; approval required")
    return peak


def bounded():
    started = time.perf_counter()

    def check(*_):
        guard()
        if time.perf_counter() - started > PROTOCOL["budget_seconds"]:
            raise TimeoutError("20-minute stage budget exceeded; approval required")

    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, 1.0, 1.0)


def clean(frame, ends):
    return frame[
        (~frame.target_imputed)
        & np.isfinite(frame.y)
        & np.isfinite(frame.yhat)
        & (frame.target_time < frame.fold.map(ends))
    ].copy()


def rolling(pred, days):
    return calibrate_online(
        pred, OnlineConfig(window=days * 24, window_hours=days * 24, min_scores=168), ALPHAS
    )


def choose(table):
    """Only development metrics enter selection; complete matched keys required."""
    policies = []
    for (model, h), group in table[table.alpha == 0.1].groupby(["model", "h"]):
        eligible = group[group.coverage.between(0.88, 0.92) & group.complete]
        selected = eligible.sort_values(["winkler_mw", "days"]).iloc[0] if len(eligible) else None
        policies.append(
            {
                "model": model,
                "h": int(h),
                "days": None if selected is None else int(selected.days),
                "status": "no_candidate_qualified" if selected is None else "selected",
            }
        )
    return policies


def generate(out, horizons):
    # Physically exclude primary calibration/test observations from development input.
    df = dataset.load().loc[lambda x: x.index < DEV.test_end].copy()
    predictions, timing = [], {}
    for factory in [RidgeForecaster, lambda: LightGBMForecaster(quantiles=False)]:
        models = []

        def create(f=factory, fitted=models):
            model = f()
            fitted.append(model)
            return model

        pred, elapsed = backtest.run_backtest(create, df, [DEV], horizons)
        predictions.append(pred)
        timing[elapsed["model"]] = elapsed
        with (out / f"{elapsed['model']}_development_model.pkl").open("wb") as handle:
            pickle.dump(models[0], handle)
        guard()
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(out / "development_predictions.parquet", index=False)
    return pred, timing


def provenance(out, start, extra):
    write(
        out / "run.json",
        {
            "git": tracking.git_state(),
            "versions": tracking.versions(),
            "device": "cpu",
            "processed_sha256": fingerprint(dataset.PROCESSED),
            "manifest": tracking.data_manifest(),
            "lock_sha256": fingerprint(config.ROOT / "uv.lock"),
            "source_sha256": {
                str(p.relative_to(config.ROOT)): fingerprint(p) for p in (config.ROOT / "src").rglob("*.py")
            },
            "wall_seconds": time.perf_counter() - start,
            "peak_rss_bytes": guard(),
            **extra,
        },
    )


def development(out, pilot=False):
    bounded()
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", PROTOCOL)  # written before any outcome is computed
    baseline_record = json.loads((BASE / "run.json").read_text())
    if fingerprint(dataset.PROCESSED) != baseline_record["data"]["processed_sha256"]:
        raise ValueError("Dataset fingerprint changed")
    pred, timing = generate(out, [168] if pilot else config.HEADLINE_HORIZONS)
    tables = []
    eligible = clean(pred[pred.block == "test"], {DEV.name: DEV.test_end})
    for days in DAYS:
        intervals = clean(rolling(pred, days), {DEV.name: DEV.test_end})
        table = interval_table(intervals, ["model", "h"])
        table["days"] = days
        table["complete"] = [
            int(n) == len(eligible[(eligible.model == m) & (eligible.h == h)])
            and intervals[(intervals.model == m) & (intervals.h == h)].lo.notna().all()
            for m, h, n in zip(table.model, table.h, table.n, strict=True)
        ]
        tables.append(table)
        guard()
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out / "development_candidates.csv", index=False)
    static = clean(static_intervals(pred), {DEV.name: DEV.test_end})
    static = static[static.alpha.isin(ALPHAS)]
    interval_table(static, ["model", "h"]).to_csv(out / "development_static.csv", index=False)
    selection = choose(table)
    write(
        out / "selection.json",
        {
            "pilot": pilot,
            "policies": selection,
            "protocol_sha256": fingerprint(out / "protocol.json"),
            "development_predictions_sha256": fingerprint(out / "development_predictions.parquet"),
            "frozen_at_utc": str(pd.Timestamp.now(tz="UTC")),
        },
    )
    provenance(out, start, {"pilot": pilot, "timings": timing})
    print(
        json.dumps(
            {
                "output": str(out),
                "seconds": time.perf_counter() - start,
                "peak_rss": guard(),
                "selection": selection,
            }
        ),
        flush=True,
    )


def paired(frame):
    rows = []
    rng = np.random.default_rng(0)
    for (model, h, alpha), g in frame.groupby(["model", "h", "alpha"]):
        keys = ["fold", "origin", "target_time"]
        s = g[g.method == "conformal_static"].set_index(keys).sort_index()
        r = g[g.method == "selected_policy"].set_index(keys).reindex(s.index)
        if r.lo.isna().any() or not np.array_equal(s.y, r.y):
            raise ValueError("Unmatched interval keys")
        ds = metrics.interval_score(
            s.y.to_numpy(), r.lo.to_numpy(), r.hi.to_numpy(), alpha
        ) - metrics.interval_score(s.y.to_numpy(), s.lo.to_numpy(), s.hi.to_numpy(), alpha)
        dc = ((r.y >= r.lo) & (r.y <= r.hi)).astype(float).to_numpy() - (
            (s.y >= s.lo) & (s.y <= s.hi)
        ).astype(float).to_numpy()
        folds = s.index.get_level_values("fold")
        for days in [7, 14, 28]:
            block = days * 24
            replicates = []
            for _ in range(500):
                indices = []
                for fold in np.unique(folds):
                    ix = np.flatnonzero(folds == fold)
                    starts = rng.integers(0, max(1, len(ix) - block + 1), size=int(np.ceil(len(ix) / block)))
                    take = np.concatenate([np.arange(a, min(a + block, len(ix))) for a in starts])[: len(ix)]
                    indices.append(ix[take])
                indices = np.concatenate(indices)
                replicates.append([ds[indices].mean(), dc[indices].mean()])
            ci = np.quantile(replicates, [0.025, 0.975], axis=0)
            rows.append(
                {
                    "model": model,
                    "h": h,
                    "alpha": alpha,
                    "block_days": days,
                    "n": len(s),
                    "effective_blocks_approx": len(s) / block,
                    "winkler_difference_mw": ds.mean(),
                    "winkler_lo": ci[0, 0],
                    "winkler_hi": ci[1, 0],
                    "coverage_difference": dc.mean(),
                    "coverage_lo": ci[0, 1],
                    "coverage_hi": ci[1, 1],
                }
            )
    return pd.DataFrame(rows)


def evaluate(out, development_source=None):
    if development_source is not None:
        if (development_source / "selection.json").resolve() == (out / "selection.json").resolve():
            raise ValueError("Use a new evaluation directory")
        selected = json.loads((development_source / "selection.json").read_text())
        if selected["development_predictions_sha256"] != fingerprint(
            development_source / "development_predictions.parquet"
        ):
            raise ValueError("Frozen development predictions changed")
        out.mkdir(parents=True, exist_ok=False)
        for name in ["selection.json", "protocol.json"]:
            shutil.copy2(development_source / name, out / name)
    bounded()
    start = time.perf_counter()
    selection_path = out / "selection.json"
    frozen_hash = fingerprint(selection_path)
    selection = json.loads(selection_path.read_text())
    if selection["pilot"] or (out / "test_intervals.parquet").exists():
        raise ValueError("Cannot evaluate pilot or overwrite existing test evidence")
    if selection["protocol_sha256"] != fingerprint(out / "protocol.json"):
        raise ValueError("Frozen protocol changed")
    expected_data = json.loads((BASE / "run.json").read_text())["data"]["processed_sha256"]
    if fingerprint(dataset.PROCESSED) != expected_data:
        raise ValueError("Dataset differs from immutable benchmark")
    expected_keys = {(m, h) for m in PROTOCOL["models"] for h in config.HEADLINE_HORIZONS}
    policies = selection["policies"]
    if (
        len(policies) != len(expected_keys)
        or {(p["model"], p["h"]) for p in policies} != expected_keys
        or any(p["days"] not in (*DAYS, None) for p in policies)
    ):
        raise ValueError("Incomplete or invalid frozen policies")
    shutil.copy2(Path(__file__), out / "executed_test_runner.py")
    shutil.copy2(config.ROOT / "src/gridcast/probabilistic/online.py", out / "executed_online.py")
    originals = {p.name: fingerprint(p) for p in BASE.glob("*.parquet")}
    frames = []
    ends = {f.name: f.test_end for f in config.TEST_FOLDS}
    for model in ["ridge", "lightgbm"]:
        pred = pd.read_parquet(BASE / f"{model}.parquet")
        static = clean(static_intervals(pred), ends)
        static = static[static.alpha.isin(ALPHAS)]
        frames.append(static)
        for policy in selection["policies"]:
            if policy["model"] != model:
                continue
            h = policy["h"]
            if policy["days"] is None:
                selected = static[static.h == h].copy()
            else:
                selected = clean(rolling(pred[pred.h == h], policy["days"]), ends)
            selected["method"] = "selected_policy"
            selected["selected_days"] = policy["days"]
            frames.append(selected)
        guard()
    intervals = pd.concat(frames, ignore_index=True)
    guard()
    intervals["season"] = season_of(pd.DatetimeIndex(intervals.target_time))
    intervals.to_parquet(out / "test_intervals.parquet", index=False)
    availability = []
    matched = []
    key = ["model", "regime", "fold", "h", "alpha", "origin", "target_time"]
    static = intervals[intervals.method == "conformal_static"].set_index(key)
    selected = intervals[intervals.method == "selected_policy"].set_index(key)
    if not static.index.sort_values().equals(selected.index.sort_values()):
        raise ValueError("Unmatched issued forecast keys")
    selected = selected.reindex(static.index)
    valid = np.isfinite(selected.lo) & np.isfinite(selected.hi)
    for (model, h, alpha), g in selected.reset_index().groupby(["model", "h", "alpha"]):
        issued = np.isfinite(g.lo) & np.isfinite(g.hi)
        covered = issued & (g.y >= g.lo) & (g.y <= g.hi)
        availability.append(
            {
                "model": model,
                "h": h,
                "alpha": alpha,
                "n_targets": len(g),
                "n_issued": int(issued.sum()),
                "n_unavailable": int((~issued).sum()),
                "issuance_fraction": float(issued.mean()),
                "coverage_missing_as_failure": float(covered.mean()),
            }
        )
    pd.DataFrame(availability).to_csv(out / "test_availability.csv", index=False)
    for method in [static, selected]:
        matched.append(method.loc[valid].reset_index())
    scored = pd.concat(matched, ignore_index=True)
    for label, by in {"overall": [], "season": ["season"], "fold": ["fold"]}.items():
        interval_table(scored, ["model", "method", "h", *by]).to_csv(out / f"test_{label}.csv", index=False)
    paired(scored).to_csv(out / "paired_block_comparisons.csv", index=False)
    if frozen_hash != fingerprint(selection_path) or any(
        fingerprint(BASE / p) != h for p, h in originals.items()
    ):
        raise RuntimeError("Immutable selection or original predictions changed")
    write(
        out / "test_run.json",
        {
            "status": "completed",
            "device": "cpu",
            "versions": tracking.versions(),
            "lock_sha256": fingerprint(config.ROOT / "uv.lock"),
            "source_sha256": {
                str(p.relative_to(config.ROOT)): fingerprint(p) for p in (config.ROOT / "src").rglob("*.py")
            },
            "development_source": str(development_source or out),
            "memory_limit_bytes": ACTIVE_MEMORY_LIMIT,
            "time_limit_seconds": PROTOCOL["budget_seconds"],
            "availability_reporting": "post-failure disclosed common-issued-key analysis; missing-as-failure coverage",
            "selection_sha256": frozen_hash,
            "baseline_prediction_sha256": originals,
            "wall_seconds": time.perf_counter() - start,
            "peak_rss_bytes": guard(),
            "test_folds": [tracking.fold_dict(f) for f in config.TEST_FOLDS],
            "processed_sha256": fingerprint(dataset.PROCESSED),
            "git": tracking.git_state(),
        },
    )
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["pilot", "development", "test"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-source", type=Path)
    parser.add_argument(
        "--memory-limit-mib",
        type=int,
        choices=[1024, 2048],
        default=1024,
        help="2048 requires explicit approval; does not change frozen selection protocol",
    )
    args = parser.parse_args()
    directory = args.output or config.RESULTS / "interval_validation" / pd.Timestamp.now(tz="UTC").strftime(
        "%Y%m%dT%H%M%SZ"
    )
    ACTIVE_MEMORY_LIMIT = args.memory_limit_mib * 1024**2
    output_preexisted = directory.exists()
    try:
        if args.stage == "test":
            evaluate(directory, args.development_source)
        else:
            development(directory, pilot=args.stage == "pilot")
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if directory.exists() and not output_preexisted:
            write(
                directory / "failure.json",
                {
                    "status": "failed",
                    "error": str(exc),
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "memory_limit_bytes": ACTIVE_MEMORY_LIMIT,
                },
            )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

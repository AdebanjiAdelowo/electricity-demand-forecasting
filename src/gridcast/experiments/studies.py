"""Controlled studies around the main benchmark.

ablation    feature-group ablations with LightGBM on the four test folds
robustness  missing observations in the history that is available at the origin
lookback    lookback-window study for the N-BEATS-style network (dev fold)
size        model-size study for the N-BEATS-style network (dev fold)
stress      Winter Storm Uri fold (January to March 2021), no NWP archive exists

Each study writes `results/tables/study_<name>.csv`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from gridcast import config, tracking
from gridcast.data import dataset, quality
from gridcast.evaluation import backtest, metrics
from gridcast.experiments import zoo
from gridcast.features.build import FeatureSpec, target_frame
from gridcast.models.base import fold_origins
from gridcast.models.dshw import DoubleSeasonalHW
from gridcast.models.naive import SeasonalNaive
from gridcast.models.tabular import LightGBMForecaster, RidgeForecaster

H = config.HEADLINE_HORIZONS
FULL = FeatureSpec(regime="oracle")

ABLATIONS: dict[str, FeatureSpec] = {
    "full": FULL,
    "weather: none": replace(FULL, regime="none"),
    "weather: temperature only": replace(FULL, weather_vars="temp"),
    "calendar: no holidays": replace(FULL, holidays=False),
    "calendar: none": replace(FULL, calendar=False),
    "lags: none (calendar + weather only)": replace(FULL, lags=False),
    "lags + calendar only, lookback 24 h": replace(FULL, regime="none", lookback=24),
    "lookback 24 h": replace(FULL, lookback=24),
    "lookback 48 h": replace(FULL, lookback=48),
    "lookback 168 h": replace(FULL, lookback=168),
    "lookback 336 h (full)": replace(FULL, lookback=337),
}


def _score(pred: pd.DataFrame, df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    scale = backtest.mase_scale(df, pred.loc[pred.block == "test", "origin"].min())
    return metrics.score_frame(backtest.scoreable(pred), scale, by)


def ablation() -> pd.DataFrame:
    df = dataset.load()
    rows = []
    with tracking.Run("study_ablation", {"variants": list(ABLATIONS)}, config.TEST_FOLDS) as run:
        for name, spec in ABLATIONS.items():
            print(f"[ablation] {name}", flush=True)
            pred, _ = backtest.run_backtest(
                lambda s=spec: LightGBMForecaster(s, "lgbm", quantiles=False),
                df,
                config.TEST_FOLDS,
                H,
                verbose=False,
            )
            t = _score(pred, df, ["regime", "h"])
            t.insert(0, "variant", name)
            rows.append(t)
        out = pd.concat(rows, ignore_index=True)
        base = out[out.variant == "full"].set_index(["regime", "h"])["mae"]
        ref = out.apply(lambda r: base.get((r["regime"], r["h"]), base.get(("forecast", r["h"]))), axis=1)
        out["mae_change_vs_full_pct"] = 100 * (out["mae"] / ref - 1)
        out.to_csv(config.TABLES / "study_ablation.csv", index=False)
        run.log(rows=len(out))
    return out


def _mask_series(df: pd.DataFrame, start, end, rate: float, block: int, seed: int) -> pd.DataFrame:
    """Remove a share of observations in [start, end) and repair them causally, as the live pipeline would."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero((df.index >= start) & (df.index < end))
    bad = np.zeros(len(df), dtype=bool)
    if rate > 0:
        n_blocks = int(round(rate * len(idx) / block))
        starts = rng.choice(idx[:-block], size=n_blocks, replace=False)
        for s in starts:
            bad[s : s + block] = True
    out = df.copy()
    out["demand"] = quality.causal_seasonal_fill(df["demand"], pd.Series(bad, index=df.index))
    out["input_missing"] = bad
    return out


def robustness(seed: int = 0) -> pd.DataFrame:
    """Train on clean data, then forecast from a history with missing hours (fold test1).

    Targets are always scored against the true series. Two patterns: isolated
    missing hours and 24-hour telemetry outages.
    """
    df = dataset.load()
    fold = config.TEST_FOLDS[0]
    models = {
        "seasonal_naive_week": lambda: SeasonalNaive(168),
        "dshw": DoubleSeasonalHW,
        "ridge_weather": lambda: RidgeForecaster(FULL, "ridge_weather"),
        "lightgbm_weather": lambda: LightGBMForecaster(FULL, "lightgbm_weather", quantiles=False),
        "nbeats_weather": zoo.deep()["nbeats_weather"],
    }
    scenarios = (
        [("none", 0.0, 1)]
        + [("isolated hours", r, 1) for r in (0.01, 0.05, 0.20)]
        + [("24 h outages", r, 24) for r in (0.05, 0.20)]
    )
    truth = target_frame(df, H)
    origins = fold_origins(df, fold)
    origins = origins[origins >= fold.test_start]
    rows = []
    with tracking.Run("study_robustness", {"scenarios": scenarios}, [fold], seed) as run:
        for name, factory in models.items():
            m = factory()
            m.fit(df, fold, H)
            print(f"[robustness] {name} fitted", flush=True)
            for pattern, rate, block in scenarios:
                dirty = _mask_series(df, fold.cal_start, fold.test_end, rate, block, seed)
                if hasattr(m, "arr"):
                    from gridcast.models.deep import Arrays

                    m.arr = Arrays(dirty, m.horizons, m.cfg.lookback, m.cfg.weather, m.arr.device)
                regime = "forecast" if "forecast" in m.regimes else "none"
                p = m.predict(dirty, origins, H, regime).merge(truth, on=["origin", "h"])
                p = p[~p.target_imputed & p.y.notna()]
                for h, g in p.groupby("h"):
                    rows.append(
                        {
                            "model": name,
                            "pattern": pattern,
                            "missing_rate": rate,
                            "h": h,
                            "mae": float((g.y - g.yhat).abs().mean()),
                            "n": len(g),
                        }
                    )
        out = pd.DataFrame(rows)
        base = out[out.pattern == "none"].set_index(["model", "h"])["mae"]
        out["mae_increase_pct"] = [100 * (r.mae / base[(r.model, r.h)] - 1) for r in out.itertuples()]
        out.to_csv(config.TABLES / "study_robustness.csv", index=False)
        run.log(rows=len(out))
    return out


def _deep_study(name: str, variants: dict[str, dict], arch: str = "nbeats") -> pd.DataFrame:
    from gridcast.models.deep import DeepForecaster, TrainConfig

    df = dataset.load()
    fold = config.DEV_FOLD
    rows = []
    with tracking.Run(f"study_{name}", {"variants": variants, "arch": arch}, [fold]) as run:
        for label, kw in variants.items():
            cfg = TrainConfig(arch=arch, weather=True, **kw)
            pred, timing = backtest.run_backtest(
                lambda c=cfg, n=f"{arch}_{label}": DeepForecaster(c, n),
                df,
                [fold],
                config.DENSE_HORIZONS,
                regimes=["oracle"],
                verbose=True,
            )
            t = _score(pred[pred.h.isin(H)], df, ["h"])
            t.insert(0, "variant", label)
            t["arch"] = arch
            t["n_params"] = timing["n_params"]
            t["fit_seconds"] = timing["fit_seconds_per_fold"]
            t["epochs_run"] = timing["extra"][fold.name]["epochs_run"]
            t["best_val_pinball"] = timing["extra"][fold.name]["best_val_pinball"]
            rows.append(t)
            pd.concat(rows, ignore_index=True).to_csv(config.TABLES / f"study_{name}.csv", index=False)
        run.log(rows=len(rows))
    return pd.concat(rows, ignore_index=True)


SIZES = {
    "nbeats": {f"width {w}": {"hidden": w, "layers": 4} for w in (32, 64, 128, 256, 512)},
    "gru": {f"hidden {w}": {"hidden": w, "layers": 2} for w in (32, 64, 128, 256)},
    "tcn": {f"channels {w}": {"hidden": w, "layers": 6} for w in (32, 64, 128)},
}


def lookback(arch: str = "nbeats") -> pd.DataFrame:
    name = "lookback" if arch == "nbeats" else f"lookback_{arch}"
    return _deep_study(name, {f"{L} h": {"lookback": L} for L in (24, 72, 168, 336)}, arch)


def size(arch: str = "nbeats") -> pd.DataFrame:
    name = "size" if arch == "nbeats" else f"size_{arch}"
    return _deep_study(name, SIZES[arch], arch)


def seeds(arch: str = "nbeats", n: int = 5) -> pd.DataFrame:
    """Seed-to-seed variation of one architecture (dev fold)."""
    return _deep_study(f"seeds_{arch}", {f"seed {i}": {"seed": i} for i in range(n)}, arch)


def stress() -> pd.DataFrame:
    """Winter Storm Uri: train on data through late 2020, forecast January to March 2021."""
    df = dataset.load()
    fold = config.URI_FOLD
    models = {
        k: v
        for k, v in zoo.classical().items()
        if k in ("seasonal_naive_week", "dshw", "ridge", "ridge_weather", "lightgbm", "lightgbm_weather")
    }
    models["lightgbm"] = lambda: LightGBMForecaster(zoo.N, "lightgbm", quantile_horizons=H)
    models["lightgbm_weather"] = lambda: LightGBMForecaster(zoo.W, "lightgbm_weather", quantile_horizons=H)
    models["nbeats_weather"] = zoo.deep()["nbeats_weather"]
    windows = {
        "before (1 Jan to 9 Feb)": ("2021-01-01", "2021-02-10"),
        "storm (10 to 20 Feb)": ("2021-02-10", "2021-02-21"),
        "after (21 Feb to 31 Mar)": ("2021-02-21", "2021-04-01"),
    }
    rows = []
    with tracking.Run("study_stress_uri", {"models": list(models), "windows": windows}, [fold]) as run:
        for name, factory in models.items():
            pred, _ = backtest.run_backtest(factory, df, [fold], H, regimes=["none", "oracle"], verbose=True)
            backtest.save_predictions(pred, "stress_uri", name)
            s = backtest.scoreable(pred)
            for label, (a, b) in windows.items():
                tt = s["target_time"]
                g = s[
                    (tt >= pd.Timestamp(a, tz=config.LOCAL_TZ)) & (tt < pd.Timestamp(b, tz=config.LOCAL_TZ))
                ]
                t = metrics.score_frame(g, None, ["model", "regime", "h"])
                t.insert(0, "window", label)
                rows.append(t)
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(config.TABLES / "study_stress_uri.csv", index=False)
        run.log(rows=len(out))
    return out


STUDIES = {
    "ablation": ablation,
    "robustness": robustness,
    "lookback": lookback,
    "size": size,
    "seeds": seeds,
    "stress": stress,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=list(STUDIES))
    ap.add_argument(
        "--arch", choices=list(SIZES), default="nbeats", help="architecture for lookback / size / seeds"
    )
    a = ap.parse_args(argv)
    out = STUDIES[a.study](a.arch) if a.study in ("lookback", "size", "seeds") else STUDIES[a.study]()
    pd.set_option("display.width", 220, "display.max_rows", 200)
    idx = [c for c in ("window", "variant", "model", "pattern", "missing_rate", "regime") if c in out.columns]
    print(out.pivot_table(index=idx, columns="h", values="mae", sort=False).round(0))


if __name__ == "__main__":
    main()

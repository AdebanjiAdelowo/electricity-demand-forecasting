"""Expanding-window rolling-origin backtest: run, store, load and score forecasts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.config import Fold
from gridcast.evaluation import metrics
from gridcast.features.build import target_frame
from gridcast.features.calendar import season_of
from gridcast.models.base import Forecaster


def attach_targets(pred: pd.DataFrame, df: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Join actuals and label each row as calibration or test for its fold."""
    tf = target_frame(df, sorted(pred["h"].unique()))
    out = pred.merge(tf, on=["origin", "h"], how="left")
    starts = {f.name: f.test_start for f in folds}
    ts = out["fold"].map(starts)
    out["block"] = np.where(out["origin"] >= ts, "test", "cal")
    # a calibration residual only exists once its target has been observed before the test block
    out.loc[(out["block"] == "cal") & (out["target_time"] >= ts), "block"] = "gap"
    return out


def run_backtest(
    factory: Callable[[], Forecaster],
    df: pd.DataFrame,
    folds: list[Fold],
    horizons: list[int],
    regimes: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    preds, fit_s, pred_s, n_pred, extra = [], 0.0, 0.0, 0, {}
    model = None
    for fold in folds:
        model = factory()
        p = model.run_fold(df, fold, horizons, regimes)
        preds.append(p)
        fit_s += model.info.fit_seconds
        pred_s += model.info.predict_seconds
        n_pred += len(p)
        extra[fold.name] = model.info.extra
        if verbose:
            print(f"  {model.name:<28s} {fold.name}: fit {model.info.fit_seconds:7.1f}s  predict {model.info.predict_seconds:6.1f}s", flush=True)
    pred = pd.concat(preds, ignore_index=True)
    pred.insert(0, "model", model.name)
    pred = attach_targets(pred, df, folds)
    timing = {
        "model": model.name, "family": model.family, "strategy": model.strategy,
        "fit_seconds_per_fold": fit_s / len(folds), "predict_seconds_per_fold": pred_s / len(folds),
        "inference_ms_per_1000_forecasts": 1e6 * pred_s / max(n_pred, 1), "n_params": model.info.n_params,
        "n_folds": len(folds), "extra": extra,
    }
    return pred, timing


def save_predictions(pred: pd.DataFrame, experiment: str, name: str) -> Path:
    d = config.PREDICTIONS / experiment
    d.mkdir(parents=True, exist_ok=True)
    out = pred.copy()
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    path = d / f"{name}.parquet"
    out.to_parquet(path, index=False)
    return path


def load_predictions(experiment: str, models: list[str] | None = None) -> pd.DataFrame:
    d = config.PREDICTIONS / experiment
    files = sorted(d.glob("*.parquet"))
    if models is not None:
        files = [f for f in files if f.stem in models]
    if not files:
        raise FileNotFoundError(f"No predictions under {d}")
    out = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in out.columns:
        if out[c].dtype == np.float32:
            out[c] = out[c].astype(np.float64)
    return out


def scoreable(pred: pd.DataFrame, block: str = "test") -> pd.DataFrame:
    """Rows that count: the requested block, a real (non-imputed) target and a finite forecast."""
    m = (pred["block"] == block) & ~pred["target_imputed"] & pred["y"].notna() & pred["yhat"].notna()
    return pred[m]


def add_slices(pred: pd.DataFrame, regime_edges: dict | None = None) -> pd.DataFrame:
    """Add season (of the target hour, local time) and load regime of the realised target."""
    out = pred.copy()
    tt = pd.DatetimeIndex(out["target_time"])
    out["season"] = season_of(tt)
    edges = regime_edges or load_regime_edges(out)
    out["load_regime"] = pd.cut(out["y"], [-np.inf, edges["q33"], edges["q67"], edges["q95"], np.inf],
                                labels=["low", "mid", "high", "peak_top5pct"]).astype(str)
    return out


def load_regime_edges(pred: pd.DataFrame) -> dict:
    y = pred.drop_duplicates("target_time")["y"].dropna()
    return {"q33": float(y.quantile(1 / 3)), "q67": float(y.quantile(2 / 3)), "q95": float(y.quantile(0.95))}


def mase_scale(df: pd.DataFrame, before: pd.Timestamp) -> float:
    return metrics.naive_scale(df.loc[df.index < before, "demand"].to_numpy())


def daily_peak_errors(pred: pd.DataFrame) -> pd.DataFrame:
    """Error of the forecast for the hour in which the realised local-day peak occurred.

    For each model/regime/horizon, keeps the forecast whose target hour is the
    daily maximum of actual demand and reports MAE and signed bias there.
    """
    p = pred.copy()
    local = (pd.DatetimeIndex(p["target_time"]) - pd.Timedelta(hours=1)).tz_convert(config.LOCAL_TZ)
    p["local_date"] = local.date
    peak = p.groupby(["model", "regime", "h", "local_date"], observed=True)["y"].transform("max")
    pk = p[p["y"] == peak]
    g = pk.groupby(["model", "regime", "h"], observed=True)
    return pd.DataFrame({
        "n_days": g.size(),
        "peak_hour_mae": g.apply(lambda d: float(np.mean(np.abs(d["y"] - d["yhat"])))),
        "peak_hour_bias": g.apply(lambda d: float(np.mean(d["yhat"] - d["y"]))),
        "peak_hour_mape": g.apply(lambda d: float(100 * np.mean(np.abs(d["y"] - d["yhat"]) / d["y"]))),
    }).reset_index()


def paired_bootstrap(pred: pd.DataFrame, baseline: str, block: int = 168, n_boot: int = 500) -> pd.DataFrame:
    """Block-bootstrap CI for MAE and for the MAE difference to `baseline` on common target hours."""
    rows = []
    key = ["regime", "h"]
    pred = pred.assign(ae=(pred["y"] - pred["yhat"]).abs())
    base = pred[pred["model"] == baseline].set_index(["h", "origin"])["ae"]
    for (model, regime, h), g in pred.groupby(["model", *key], observed=True):
        g = g.sort_values("origin")
        lo, hi = metrics.block_bootstrap_ci(g["ae"].to_numpy(), block, n_boot)
        rec = {"model": model, "regime": regime, "h": h, "mae": g["ae"].mean(), "mae_lo": lo, "mae_hi": hi}
        b = base.reindex(pd.MultiIndex.from_arrays([g["h"], g["origin"]]))
        d = g["ae"].to_numpy() - b.to_numpy()
        ok = np.isfinite(d)
        if ok.sum() > 2 * block and model != baseline:
            dl, dh = metrics.block_bootstrap_ci(d[ok], block, n_boot)
            rec.update({"diff_vs_baseline": float(d[ok].mean()), "diff_lo": dl, "diff_hi": dh,
                        "significant": bool(dl > 0 or dh < 0)})
        rows.append(rec)
    return pd.DataFrame(rows)

"""Descriptive distribution-shift diagnostics; no IID significance claims."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from gridcast.features.calendar import local_start, season_of


def slices(pred: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = pred.copy()
    idx = pd.DatetimeIndex(out.target_time)
    local = local_start(idx)
    out["season"] = season_of(idx)
    out["month"] = local.month
    out["year_month"] = local.strftime("%Y-%m")
    out["day_night"] = np.where((local.hour >= 6) & (local.hour < 18), "day_06_18", "night_18_06")
    out["week_part"] = np.where(local.dayofweek < 5, "weekday", "weekend")
    out["demand_regime"] = ""
    for fold, edges in thresholds.items():
        keep = out.fold == fold
        out.loc[keep, "demand_regime"] = pd.cut(out.loc[keep, "y"], [-np.inf, *edges, np.inf],
                                                labels=["low", "mid", "high", "peak_top5pct"]).astype(str)
    return out


def residual_drift(pred: pd.DataFrame, scales: dict) -> pd.DataFrame:
    rows = []
    for (model, regime, h), g in pred.groupby(["model", "regime", "h"], observed=True):
        folds = sorted(g.fold.unique())
        for first, second in zip(folds[:-1], folds[1:], strict=True):
            a, b = g[g.fold == first], g[g.fold == second]
            ea, eb = (a.y - a.yhat).to_numpy(float), (b.y - b.yhat).to_numpy(float)
            sa, sb = ea / scales[first], eb / scales[second]
            rows.append({"model": model, "regime": regime, "h": h, "from_fold": first, "to_fold": second,
                         "n_from": len(a), "n_to": len(b), "mae_change_mw": abs(eb).mean() - abs(ea).mean(),
                         "bias_change_mw": eb.mean() - ea.mean(), "residual_wasserstein_mw": wasserstein_distance(ea, eb),
                         "scaled_residual_wasserstein": wasserstein_distance(sa, sb),
                         "ks_statistic_descriptive": ks_2samp(sa, sb).statistic})
    return pd.DataFrame(rows)

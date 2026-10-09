"""Conformal calibration of prediction intervals.

Works on scored prediction frames (see `evaluation.backtest.attach_targets`).
Two families, each in a static and a rolling form:

* absolute-residual split conformal around a point forecast;
* conformalised quantile regression (CQR, Romano et al. 2019) on top of
  model quantiles.

Static calibration uses the fold's calibration block only (residuals whose
targets were observed before the test block starts).

Rolling calibration is an online scheme: at origin t and horizon h it uses the
scores of the most recent `window` forecasts whose targets were already
observed, that is forecasts issued at or before t - h. It adapts to seasonal
drift but, like any time-series conformal scheme, loses the exchangeability
guarantee; its coverage is an empirical finding, not a theorem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.evaluation.metrics import qcols

KEYS = ["model", "regime", "fold", "h"]


def _finite_sample_level(n: int, alpha: float) -> float:
    return float(min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))


def _levels(alpha: float) -> tuple[str, str]:
    return f"q{alpha / 2:g}", f"q{1 - alpha / 2:g}"


def _score(g: pd.DataFrame, alpha: float, kind: str) -> np.ndarray:
    if kind == "abs":
        return (g["y"] - g["yhat"]).abs().to_numpy(float)
    lo, hi = _levels(alpha)
    return np.maximum(g[lo] - g["y"], g["y"] - g[hi]).to_numpy(float)


def _apply(g: pd.DataFrame, alpha: float, kind: str, margin) -> tuple[np.ndarray, np.ndarray]:
    if kind == "abs":
        return g["yhat"].to_numpy(float) - margin, g["yhat"].to_numpy(float) + margin
    lo, hi = _levels(alpha)
    return g[lo].to_numpy(float) - margin, g[hi].to_numpy(float) + margin


def calibrate(pred: pd.DataFrame, kind: str = "abs", mode: str = "static", window: int = 24 * 60,
              min_scores: int = 24 * 7) -> pd.DataFrame:
    """Return test-block rows with calibrated quantile columns.

    kind : "abs" (point forecast +- residual quantile) or "cqr" (shift model quantiles)
    mode : "static" (fold calibration block) or "rolling" (trailing window, online)
    """
    out = []
    for _, g in pred.groupby(KEYS, observed=True, sort=False):
        g = g.sort_values("origin")
        h = int(g["h"].iloc[0])
        valid = (~g["target_imputed"] & g["y"].notna() & g["yhat"].notna()).to_numpy()
        test = (g["block"] == "test").to_numpy()
        res = g.loc[test].copy()
        if kind == "abs":
            res["q0.5"] = res["yhat"]
        for alpha in config.INTERVAL_ALPHAS:
            s = np.where(valid, _score(g, alpha, kind), np.nan)
            if mode == "static":
                cal = s[(g["block"] == "cal").to_numpy() & valid]
                margin = np.quantile(cal, _finite_sample_level(len(cal), alpha), method="higher")
            else:
                level = _finite_sample_level(window, alpha)
                roll = pd.Series(s).rolling(window, min_periods=min_scores).quantile(level, interpolation="higher")
                margin = roll.shift(h).to_numpy()[test]  # only scores whose target is already observed
            lo, hi = _apply(g.loc[test], alpha, kind, margin)
            a, b = _levels(alpha)
            res[a], res[b] = lo, hi
        q = np.sort(res[qcols()].to_numpy(float), axis=1)
        res[qcols()] = q
        out.append(res)
    res = pd.concat(out, ignore_index=True)
    res["method"] = f"{'conformal' if kind == 'abs' else 'cqr'}_{mode}"
    return res


def reliability(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Empirical frequency of y <= q for every quantile level (PIT reliability)."""
    rows = []
    for key, g in pred.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        for lv, c in zip(config.QUANTILES, qcols(), strict=True):
            rows.append({**dict(zip(by, key, strict=True)), "nominal": lv, "empirical": float((g["y"] <= g[c]).mean())})
    return pd.DataFrame(rows)

"""Point and probabilistic forecast metrics.

All functions take plain numpy arrays. Quantile forecasts are passed as a 2-D
array `q` of shape (n, len(levels)) with `levels` ascending and containing 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast import config


def mae(y, f):
    return float(np.mean(np.abs(y - f)))


def rmse(y, f):
    return float(np.sqrt(np.mean((y - f) ** 2)))


def bias(y, f):
    return float(np.mean(f - y))


def mape(y, f):
    return float(100 * np.mean(np.abs(y - f) / np.abs(y)))


def smape(y, f):
    return float(100 * np.mean(2 * np.abs(y - f) / (np.abs(y) + np.abs(f))))


def naive_scale(train: np.ndarray, period: int = config.SEASONAL_PERIOD) -> float:
    """In-sample MAE of the seasonal naive forecast: the MASE denominator."""
    train = np.asarray(train, dtype=float)
    d = np.abs(train[period:] - train[:-period])
    return float(np.nanmean(d))


def mase(y, f, scale: float):
    return mae(y, f) / scale


def pinball(y, q, levels) -> float:
    y = np.asarray(y)[:, None]
    lv = np.asarray(levels)[None, :]
    d = y - q
    return float(np.mean(np.maximum(lv * d, (lv - 1) * d)))


def interval_stats(y, q, levels, alpha: float) -> tuple[float, float]:
    """Empirical coverage and mean width of the central (1 - alpha) interval."""
    levels = list(np.round(levels, 6))
    lo = q[:, levels.index(round(alpha / 2, 6))]
    hi = q[:, levels.index(round(1 - alpha / 2, 6))]
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def interval_score(y, lo, hi, alpha):
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def wis(y, q, levels, alphas=config.INTERVAL_ALPHAS) -> float:
    """Weighted interval score (Bracher et al. 2021) from K central intervals and the median."""
    levels = list(np.round(levels, 6))
    med = q[:, levels.index(0.5)]
    total = 0.5 * np.abs(y - med)
    for a in alphas:
        lo = q[:, levels.index(round(a / 2, 6))]
        hi = q[:, levels.index(round(1 - a / 2, 6))]
        total = total + (a / 2) * interval_score(y, lo, hi, a)
    return float(np.mean(total / (len(alphas) + 0.5)))


def crps_from_quantiles(y, q, levels) -> float:
    """CRPS approximated as twice the pinball loss integrated over the quantile grid (trapezoid)."""
    y = np.asarray(y)[:, None]
    lv = np.asarray(levels)
    d = y - q
    pl = np.maximum(lv * d, (lv - 1) * d)  # (n, k)
    return float(np.mean(2 * np.trapezoid(pl, lv, axis=1) / (lv[-1] - lv[0])))


def point_metrics(y, f, scale: float | None = None) -> dict:
    out = {"n": int(len(y)), "mae": mae(y, f), "rmse": rmse(y, f), "mape": mape(y, f),
           "smape": smape(y, f), "bias": bias(y, f)}
    if scale:
        out["mase"] = mase(y, f, scale)
    return out


def prob_metrics(y, q, levels) -> dict:
    out = {"pinball": pinball(y, q, levels), "wis": wis(y, q, levels), "crps": crps_from_quantiles(y, q, levels)}
    for a in config.INTERVAL_ALPHAS:
        cov, width = interval_stats(y, q, levels, a)
        out[f"cov{int(round((1 - a) * 100))}"] = cov
        out[f"width{int(round((1 - a) * 100))}"] = width
    return out


def block_bootstrap_ci(values: np.ndarray, block: int = 168, n_boot: int = 500, seed: int = 0, level: float = 0.95):
    """Moving-block bootstrap CI for the mean of a serially dependent sequence (ordered in time)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2 * block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    csum = np.concatenate([[0.0], np.cumsum(v)])
    sums = csum[starts + block] - csum[starts]
    means = sums.sum(axis=1) / (n_blocks * block)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def qcols(levels=config.QUANTILES) -> list[str]:
    return [f"q{lv:g}" for lv in levels]


def score_frame(df: pd.DataFrame, scale: float | None, by: list[str]) -> pd.DataFrame:
    """Score a long prediction frame with columns y, yhat and optional quantile columns."""
    cols = [c for c in qcols() if c in df.columns]
    rows = []
    for key, g in df.groupby(by, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        y = g["y"].to_numpy(float)
        rec = dict(zip(by, key, strict=True))
        rec.update(point_metrics(y, g["yhat"].to_numpy(float), scale))
        if len(cols) == len(config.QUANTILES) and g[cols].notna().all().all():
            rec.update(prob_metrics(y, g[cols].to_numpy(float), config.QUANTILES))
        rows.append(rec)
    return pd.DataFrame(rows)

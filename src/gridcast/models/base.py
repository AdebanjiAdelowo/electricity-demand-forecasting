"""Common forecaster interface.

A forecaster is fit once per backtest fold and returns a long frame of
forecasts for every hourly origin in the fold's calibration and test blocks:

    origin (UTC, hour ending) | h | regime | yhat | [q0.025 ... q0.975]

`regime` records which weather information the forecast used. Forecasters
never see the target frame; scoring happens elsewhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.config import Fold
from gridcast.evaluation.metrics import qcols


@dataclass
class FitInfo:
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    n_params: int | None = None
    extra: dict = field(default_factory=dict)


class Forecaster:
    name: str = "base"
    family: str = "statistical"  # statistical | ml | deep | foundation
    strategy: str = "direct"  # direct | recursive | multi-output
    regimes: tuple[str, ...] = ("none",)
    probabilistic: bool = False

    def __init__(self) -> None:
        self.info = FitInfo()

    def fit(self, df: pd.DataFrame, fold: Fold, horizons: list[int]) -> None:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame, origins: pd.DatetimeIndex, horizons: list[int], regime: str) -> pd.DataFrame:
        raise NotImplementedError

    def run_fold(self, df: pd.DataFrame, fold: Fold, horizons: list[int], regimes: list[str] | None = None) -> pd.DataFrame:
        regimes = [r for r in (regimes or self.regimes) if r in self.regimes]
        t0 = time.perf_counter()
        self.fit(df, fold, horizons)
        self.info.fit_seconds = time.perf_counter() - t0
        origins = fold_origins(df, fold)
        t0 = time.perf_counter()
        out = [self.predict(df, origins, horizons, r) for r in regimes]
        self.info.predict_seconds = time.perf_counter() - t0
        out = pd.concat(out, ignore_index=True)
        out["fold"] = fold.name
        return out


def fold_origins(df: pd.DataFrame, fold: Fold) -> pd.DatetimeIndex:
    return df.index[(df.index >= fold.cal_start) & (df.index < fold.test_end)]


def fit_slice(df: pd.DataFrame, fold: Fold, include_val: bool = True) -> pd.DataFrame:
    """Rows whose timestamp is strictly inside the fitting period."""
    end = fold.cal_start if include_val else fold.val_start
    return df[(df.index >= fold.train_start) & (df.index < end)]


def train_mask(index: pd.DatetimeIndex, h: int, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Origins whose *target* also lies before `end`, so no label crosses the boundary."""
    return np.asarray((index >= start) & (index + pd.Timedelta(hours=h) < end))


def long_frame(origins: pd.DatetimeIndex, h: int, regime: str, yhat: np.ndarray, q: np.ndarray | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"origin": origins, "h": h, "regime": regime, "yhat": np.asarray(yhat, dtype=float)})
    if q is not None:
        q = np.sort(np.asarray(q, dtype=float), axis=1)  # remove quantile crossing
        for i, c in enumerate(qcols(config.QUANTILES)):
            out[c] = q[:, i]
    return out

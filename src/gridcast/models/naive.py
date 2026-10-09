"""Reference forecasts that need no estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.models.base import Forecaster, long_frame


class _Shift(Forecaster):
    family = "statistical"
    strategy = "direct"

    def fit(self, df, fold, horizons):
        return None

    def _series(self, df: pd.DataFrame, h: int) -> pd.Series:
        raise NotImplementedError

    def predict(self, df, origins, horizons, regime):
        return pd.concat(
            [long_frame(origins, h, regime, self._series(df, h).reindex(origins).to_numpy()) for h in horizons],
            ignore_index=True,
        )


class Persistence(_Shift):
    """Last observed value."""

    name = "persistence"

    def _series(self, df, h):
        return df["demand"]


class SeasonalNaive(_Shift):
    """Most recent observed value at the same position in the seasonal cycle."""

    def __init__(self, period: int = 168):
        super().__init__()
        self.period = period
        self.name = "seasonal_naive_week" if period == 168 else f"seasonal_naive_{period}h"

    def _series(self, df, h):
        k = -(-h // self.period)
        return df["demand"].shift(self.period * k - h)


class SeasonalProfile(_Shift):
    """Mean of the last `n_weeks` observed values at the target's hour of week."""

    def __init__(self, n_weeks: int = 4):
        super().__init__()
        self.n_weeks = n_weeks
        self.name = f"seasonal_profile_{n_weeks}w"

    def _series(self, df, h):
        k = -(-h // 168)
        y = df["demand"]
        return sum(y.shift(168 * (k + i) - h) for i in range(self.n_weeks)) / self.n_weeks


class EiaDayAhead(Forecaster):
    """ERCOT's own day-ahead demand forecast as published in EIA-930.

    This is an external reference, not a model of this study. It is only
    defined for the target hour, its true issue time is roughly the morning of
    the previous day, and it is reported next to the 24 h horizon for context.
    """

    name = "ercot_day_ahead_reference"
    family = "external"

    def fit(self, df, fold, horizons):
        return None

    def predict(self, df, origins, horizons, regime):
        frames = []
        for h in horizons:
            f = df["eia_day_ahead_forecast"].shift(-h).reindex(origins).to_numpy()
            frames.append(long_frame(origins, h, regime, np.asarray(f, dtype=float)))
        return pd.concat(frames, ignore_index=True)

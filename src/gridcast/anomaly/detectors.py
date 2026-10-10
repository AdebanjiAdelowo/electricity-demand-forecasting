"""Unsupervised anomaly scores for hourly demand. Higher score = more unusual.

Every detector is causal: the score at hour t uses observations up to t only,
so it could run online. None of them is trained on anomaly labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from gridcast.features.build import FeatureSpec, make_features
from gridcast.features.calendar import calendar_frame

MAD_K = 1.4826


def robust_z(resid: pd.Series, window: int = 24 * 28, min_periods: int = 24 * 7) -> pd.Series:
    """Residual standardised by trailing median and MAD (excluding the current hour)."""
    past = resid.shift(1)
    med = past.rolling(window, min_periods=min_periods).median()
    mad = (past - med).abs().rolling(window, min_periods=min_periods).median()
    return (resid - med) / (MAD_K * mad).replace(0, np.nan)


def seasonal_residual(y: pd.Series, n_weeks: int = 4) -> pd.Series:
    """Relative deviation from the median of the same hour of week over the previous n weeks."""
    ref = pd.concat([y.shift(168 * k) for k in range(1, n_weeks + 1)], axis=1).median(axis=1)
    return (y - ref) / ref


def score_robust_seasonal(df: pd.DataFrame) -> pd.Series:
    return robust_z(seasonal_residual(df["demand"])).abs().rename("robust_seasonal")


class ForecastResidual:
    """Residual of a one-step-style forecast, standardised by its trailing robust scale.

    The forecaster is fit on clean history. At detection time the actual
    weather for hour t is already observed, so using reanalysis weather here is
    legitimate (this is monitoring, not forecasting).
    """

    def __init__(self, model, h: int = 1, regime: str = "none"):
        self.model, self.h, self.regime = model, h, regime
        self.name = f"forecast_residual_h{h}"

    def residual(self, df: pd.DataFrame) -> pd.Series:
        pred = self.model.predict(df, df.index, [self.h], self.regime)
        yhat = pd.Series(pred["yhat"].to_numpy(), index=df.index + pd.Timedelta(hours=self.h))
        yhat = yhat.reindex(df.index)
        return ((df["demand"] - yhat) / yhat).rename("resid")

    def score(self, df: pd.DataFrame) -> pd.Series:
        return robust_z(self.residual(df)).abs().rename(self.name)


def cusum(z: pd.Series, k: float = 0.5, cap: float = 50.0) -> pd.Series:
    """Two-sided CUSUM of a standardised residual; the score is max(S+, S-).

    Accumulates persistent small deviations, so it reacts to level shifts that
    never produce a single large residual. `cap` bounds the statistic so that
    it recovers after an event.
    """
    v = np.nan_to_num(z.to_numpy(float), nan=0.0)
    v = np.clip(v, -8, 8)
    sp = np.zeros(len(v))
    sn = np.zeros(len(v))
    a = b = 0.0
    for i, x in enumerate(v):
        a = min(max(0.0, a + x - k), cap)
        b = min(max(0.0, b - x - k), cap)
        sp[i], sn[i] = a, b
    return pd.Series(np.maximum(sp, sn), index=z.index, name="cusum")


def flatline_score(y: pd.Series, max_run: int = 24) -> pd.Series:
    """Length of the current run of (near-)identical values: catches stuck telemetry."""
    same = (y.diff().abs() < 1e-9 * y.abs().clip(lower=1)).astype(int)
    run = same.groupby((same == 0).cumsum()).cumsum()
    return run.clip(upper=max_run).astype(float).rename("flatline")


def detector_features(df: pd.DataFrame, fr_resid: pd.Series | None = None) -> pd.DataFrame:
    y = df["demand"]
    cal = calendar_frame(df.index)
    f = pd.DataFrame({
        "seasonal_resid": seasonal_residual(y),
        "rel_diff1": y.pct_change(1),
        "rel_diff24": y.pct_change(24),
        "rel_diff168": y.pct_change(168),
        "roll_std6": y.rolling(6).std() / y.rolling(168).mean(),
        "hour_sin": cal["hour_sin"], "hour_cos": cal["hour_cos"], "is_offday": cal["is_offday"],
    })
    if fr_resid is not None:
        f["forecast_resid"] = fr_resid
    return f


class IsoForest:
    name = "isolation_forest"

    def __init__(self, seed: int = 0, n_estimators: int = 200):
        self.model = IsolationForest(n_estimators=n_estimators, contamination="auto", random_state=seed, n_jobs=-1)
        self.cols: list[str] | None = None

    def fit(self, feats: pd.DataFrame) -> IsoForest:
        f = feats.dropna()
        self.cols = list(f.columns)
        self.model.fit(f.to_numpy(float))
        return self

    def score(self, feats: pd.DataFrame) -> pd.Series:
        f = feats[self.cols]
        ok = f.notna().all(axis=1)
        s = pd.Series(np.nan, index=feats.index, name=self.name)
        s[ok] = -self.model.score_samples(f[ok].to_numpy(float))
        return s


def fit_residual_model(df: pd.DataFrame, fold, h: int = 1, weather: bool = False):
    """Small LightGBM point forecaster used only to produce monitoring residuals."""
    from gridcast.models.tabular import LightGBMForecaster

    spec = FeatureSpec(regime="oracle" if weather else "none")
    m = LightGBMForecaster(spec, f"monitor_h{h}", quantiles=False, n_estimators=300, num_leaves=31)
    m.fit(df, fold, [h])
    return m


__all__ = ["ForecastResidual", "IsoForest", "cusum", "detector_features", "fit_residual_model", "flatline_score",
           "make_features", "robust_z", "score_robust_seasonal", "seasonal_residual"]

"""Leakage-safe tabular features for direct multi-horizon forecasting.

Row t of the feature matrix for horizon h describes the forecast made at
origin t (after the demand for the hour ending at t is known) for the target
hour ending at t + h.

Availability rules enforced here:

* demand            : only values at or before t (non-negative shifts).
* calendar          : deterministic, taken at the target time.
* weather, "oracle" : reanalysis at the target time (NOT operational).
* weather, "forecast": archived NWP forecast for the target time, taken from a
                      lead bucket that was already published at t.
* weather, "none"   : no weather columns at all.
* observed weather at or before t is treated as known in both weather regimes.

Demand features and the target are divided by the trailing mean of the last
`norm_window` hours at the origin. ERCOT load grew by roughly 50 percent over
the sample, and this keeps tree and neural models inside their training range.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.features.calendar import calendar_frame

BASE_TEMP_C = 18.3  # 65 F, the conventional degree-day base
ORIGIN_LAGS = [0, 1, 2, 3, 6, 12, 23, 24, 25, 47, 48, 71, 72, 167, 168, 169, 335, 336]


@dataclass(frozen=True)
class FeatureSpec:
    regime: str = "none"  # none | forecast | oracle
    lags: bool = True
    calendar: bool = True
    holidays: bool = True
    lookback: int = 336  # hours of demand history the lag features may reach
    weather_vars: str = "full"  # full | temp
    norm_window: int = 168

    @property
    def weather(self) -> bool:
        return self.regime != "none"

    def training_view(self) -> FeatureSpec:
        """Models are always trained on reanalysis weather (no long NWP archive exists)."""
        return replace(self, regime="oracle") if self.weather else self


def level(df: pd.DataFrame, window: int = 168) -> pd.Series:
    return df["demand"].rolling(window, min_periods=window).mean()


def target_weather(df: pd.DataFrame, h: int, regime: str) -> pd.DataFrame:
    """Weather series, indexed by valid time, that horizon h may use for its target."""
    if regime == "oracle":
        cols = {"temp": "temp", "rh": "rh", "wind": "wind", "tmin": "tmin", "tmax": "tmax"}
    elif regime == "forecast":
        d = config.forecast_lead_day(h)
        cols = {k: f"{k}_d{d}" for k in ("temp", "rh", "wind", "tmin", "tmax")}
    else:
        raise ValueError(regime)
    return df[list(cols.values())].rename(columns={v: k for k, v in cols.items()})


def _weather_features(df: pd.DataFrame, h: int, spec: FeatureSpec) -> pd.DataFrame:
    w = target_weather(df, h, spec.regime)
    f = pd.DataFrame(index=df.index)
    t = w["temp"]
    f["w_temp"] = t
    f["w_cdd"] = (t - BASE_TEMP_C).clip(lower=0)
    f["w_hdd"] = (BASE_TEMP_C - t).clip(lower=0)
    f["w_temp_mean24"] = t.rolling(24, min_periods=18).mean()
    f["w_temp_mean72"] = t.rolling(72, min_periods=48).mean()
    f["w_temp_max24"] = t.rolling(24, min_periods=18).max()
    f["w_temp_min24"] = t.rolling(24, min_periods=18).min()
    if spec.weather_vars == "full":
        f["w_tmin_points"] = w["tmin"]
        f["w_tmax_points"] = w["tmax"]
        f["w_rh"] = w["rh"]
        f["w_wind"] = w["wind"]
        f["w_humid_heat"] = f["w_cdd"] * w["rh"] / 100.0
        f["w_wind_chill"] = f["w_hdd"] * w["wind"]
    f = f.shift(-h)  # known-future covariate: move target-time value to the origin row
    # observed weather at the origin, and the temperature change relative to the
    # hour that the same-hour demand lag refers to (always at or before t)
    j = -(-h // 24)
    obs = df["temp"]
    f["w_obs_temp_origin"] = obs
    f["w_obs_temp_mean24"] = obs.rolling(24, min_periods=18).mean()
    f["w_delta_vs_daylag"] = f["w_temp"] - obs.shift(24 * j - h)
    return f


def _calendar_features(df: pd.DataFrame, h: int, spec: FeatureSpec) -> pd.DataFrame:
    cal = calendar_frame(df.index)
    if not spec.holidays:
        cal = cal.drop(columns=["is_holiday", "is_holiday_adjacent"])
        cal["is_offday"] = cal["is_weekend"]
    cal = cal.drop(columns=["is_dst"]).add_prefix("c_")
    return cal.shift(-h)


def _lag_features(df: pd.DataFrame, h: int, spec: FeatureSpec, m: pd.Series) -> pd.DataFrame:
    y = df["demand"]
    f = pd.DataFrame(index=df.index)
    for lag in ORIGIN_LAGS:
        if lag < spec.lookback:
            f[f"l_lag{lag}"] = y.shift(lag) / m
    j = -(-h // 24)
    k = -(-h // 168)
    # most recent observed value at the same hour of day / hour of week as the target
    if 24 * j - h < spec.lookback:
        f["l_same_hour_day"] = y.shift(24 * j - h) / m
    if 24 * (j + 1) - h < spec.lookback:
        f["l_same_hour_day2"] = y.shift(24 * (j + 1) - h) / m
    if 168 * k - h < spec.lookback:
        f["l_same_hour_week"] = y.shift(168 * k - h) / m
    if 168 * (k + 1) - h < spec.lookback:
        f["l_same_hour_week2"] = y.shift(168 * (k + 1) - h) / m
    for win in (24, 168):
        if win <= spec.lookback:
            r = y.rolling(win, min_periods=win)
            f[f"l_mean{win}"] = r.mean() / m
            f[f"l_std{win}"] = r.std() / m
            f[f"l_min{win}"] = r.min() / m
            f[f"l_max{win}"] = r.max() / m
    if spec.lookback > 24:
        f["l_diff24"] = (y - y.shift(24)) / m
    if spec.lookback > 168:
        f["l_diff168"] = (y - y.shift(168)) / m
    return f


def make_features(df: pd.DataFrame, h: int, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y_norm, level). All are indexed by forecast origin."""
    m = level(df, min(spec.norm_window, max(spec.lookback, 24)))
    parts = []
    if spec.lags:
        parts.append(_lag_features(df, h, spec, m))
    if spec.calendar:
        parts.append(_calendar_features(df, h, spec))
    if spec.weather:
        parts.append(_weather_features(df, h, spec))
    if not parts:
        raise ValueError("FeatureSpec selects no features")
    X = pd.concat(parts, axis=1)
    y = df["demand"].shift(-h) / m
    return X, y, m


def target_frame(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Long frame of (origin, h) -> actual target, target time and scoring mask."""
    rows = []
    for h in horizons:
        rows.append(
            pd.DataFrame(
                {
                    "origin": df.index,
                    "h": h,
                    "target_time": df.index + pd.Timedelta(hours=h),
                    "y": df["demand"].shift(-h).to_numpy(),
                    "target_imputed": df["is_imputed"].shift(-h).fillna(True).to_numpy(dtype=bool),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def one_hot_design(X: pd.DataFrame) -> pd.DataFrame:
    """Expand calendar integers for linear models (hour, day of week, hour x off-day, degree days x daypart)."""
    X = X.copy()
    if "c_hour" in X:
        hour = X["c_hour"]
        off = X["c_is_offday"].fillna(0)
        for v in range(24):
            X[f"oh_hour{v}"] = (hour == v).astype(float)
            X[f"oh_hour{v}_off"] = ((hour == v) & (off == 1)).astype(float)
        for v in range(7):
            X[f"oh_dow{v}"] = (X["c_dow"] == v).astype(float)
        if "w_cdd" in X:
            part = (hour // 6).fillna(0)
            for v in range(4):
                X[f"w_cdd_part{v}"] = X["w_cdd"] * (part == v)
                X[f"w_hdd_part{v}"] = X["w_hdd"] * (part == v)
            X["w_cdd_sq"] = X["w_cdd"] ** 2
            X["w_hdd_sq"] = X["w_hdd"] ** 2
        X = X.drop(columns=["c_hour", "c_dow", "c_month"])
    return X.astype(float)


def finite_rows(X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
    ok = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    if y is not None:
        ok &= np.isfinite(y.to_numpy(dtype=float))
    return ok

"""Explicit weather availability; never infer issue times from lead-day buckets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.features.build import FeatureSpec, make_features


@dataclass(frozen=True)
class WeatherConfig:
    mode: str = "history_proxy"  # none | history_proxy | issued | oracle
    variables: tuple[str, ...] = ("temp", "rh", "wind")
    observation_delay_hours: int = 1
    lags: tuple[int, ...] = (0, 1, 24)
    rolling_hours: tuple[int, ...] = (24, 72)
    fill_limit_hours: int = 3

    def __post_init__(self):
        if self.mode not in {"none", "history_proxy", "issued", "oracle"}:
            raise ValueError("Unknown weather information mode")
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise ValueError("Weather variables must be nonempty and unique")
        if not set(self.variables) <= {"temp", "rh", "wind", "tmin", "tmax"}:
            raise ValueError("Unsupported weather variable")
        for name, value in [("observation delay", self.observation_delay_hours), ("fill limit", self.fill_limit_hours)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Invalid {name}")
        if any(not isinstance(k, int) or isinstance(k, bool) or k < 0 for k in self.lags):
            raise ValueError("Weather lags must be nonnegative integer hours")
        if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in self.rolling_hours):
            raise ValueError("Rolling windows must be positive integer hours")


def utc_times(values, name: str) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise ValueError(f"{name} must be explicit timezone-aware timestamps")
    return index.tz_convert("UTC").as_unit("ns")


def validate_archive(archive: pd.DataFrame, variables: tuple[str, ...]) -> pd.DataFrame:
    required = {"issue_time", "available_at", "valid_time", "source", *variables}
    if not required <= set(archive.columns):
        raise ValueError(f"Forecast archive missing {sorted(required - set(archive.columns))}")
    a = archive.copy()
    for col in ("issue_time", "available_at", "valid_time"):
        a[col] = utc_times(a[col], col)
    if (a.available_at < a.issue_time).any() or (a.valid_time < a.issue_time).any():
        raise ValueError("Impossible issue/publication/valid timestamp ordering")
    if a.source.isna().any() or a.source.astype(str).str.strip().eq("").any():
        raise ValueError("Forecast rows require provenance")
    if a.duplicated(["valid_time", "available_at"]).any():
        raise ValueError("Ambiguous forecast revisions at identical availability times")
    for col in variables:
        a[col] = pd.to_numeric(a[col], errors="raise")
        if np.isinf(a[col]).any():
            raise ValueError("Infinite weather values")
    return a


def select_issued(archive: pd.DataFrame, origins: pd.DatetimeIndex, valid: pd.DatetimeIndex,
                  variables: tuple[str, ...]) -> pd.DataFrame:
    """Latest row published by each origin for its requested valid time.

    Timestamp/source evidence must be supplied by the caller. Schema validation
    cannot establish the truth of a provider's publication-time claim.
    """
    origins, valid = utc_times(origins, "origins"), utc_times(valid, "valid times")
    if len(origins) != len(valid):
        raise ValueError("Origin/valid pairs differ in length")
    a = validate_archive(archive, variables)
    left = pd.DataFrame({"origin": origins, "valid_time": valid, "row": np.arange(len(origins))})
    right = a.sort_values("available_at")
    chosen = pd.merge_asof(left.sort_values("origin"), right, left_on="origin", right_on="available_at",
                           by="valid_time", direction="backward").sort_values("row")
    chosen.index = origins
    return chosen.drop(columns="row")


def features(df: pd.DataFrame, h: int, weather: WeatherConfig,
             archive: pd.DataFrame | None = None):
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError("Horizon must be a positive integer hour")
    idx = utc_times(df.index, "data index")
    if not idx.is_unique or not idx.is_monotonic_increasing:
        raise ValueError("Need ordered unique data timestamps")
    if len(idx) > 1 and not np.all(np.diff(idx.asi8) == pd.Timedelta(hours=1).value):
        raise ValueError("Demand/calendar pipeline requires a complete hourly grid")
    x, y, level = make_features(df, h, FeatureSpec())
    if weather.mode == "none":
        return x, y, level
    missing = set(weather.variables) - set(df.columns)
    if missing:
        raise ValueError(f"Observation proxy missing {sorted(missing)}")
    observed = df[list(weather.variables)].astype(float).replace([np.inf, -np.inf], np.nan)
    delayed = observed.shift(weather.observation_delay_hours)
    filled = delayed.ffill(limit=weather.fill_limit_hours) if weather.fill_limit_hours else delayed
    for variable in weather.variables:
        x[f"wx_{variable}_missing"] = delayed[variable].isna().astype(float)
        for lag in weather.lags:
            x[f"wx_{variable}_lag{lag}"] = filled[variable].shift(lag)
        for window in weather.rolling_hours:
            x[f"wx_{variable}_mean{window}"] = filled[variable].rolling(window, min_periods=window).mean()
    if weather.mode in {"oracle", "issued"}:
        for lag in (0, 1, 23):
            valid = idx + pd.Timedelta(hours=h - lag)
            if weather.mode == "oracle":
                chosen = observed.reindex(valid).set_axis(df.index)
            else:
                if archive is None:
                    raise ValueError("Issued mode requires a timestamped archive; lead buckets are insufficient")
                chosen = select_issued(archive, idx, valid, weather.variables)
            for variable in weather.variables:
                values = chosen[variable].to_numpy(float)
                x[f"wx_{variable}_target_lag{lag}"] = values
                x[f"wx_{variable}_target_lag{lag}_missing"] = np.isnan(values).astype(float)
    x.attrs["information_mode"] = weather.mode
    x.attrs["operational_availability_verified"] = False  # requires external source audit, not merely schema
    return x, y, level

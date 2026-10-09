"""Calendar features derived from UTC hour-ending timestamps.

The local label of an interval is taken from its *start* (utc_end - 1h), so the
hour ending at local midnight belongs to the day it actually covers. Daylight
saving is handled by the tz database: the spring-forward hour simply does not
occur and the fall-back hour occurs twice, exactly as in the real system.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from gridcast import config


@lru_cache(maxsize=4)
def _holiday_dates(year_min: int, year_max: int) -> frozenset:
    cal = holidays.country_holidays("US", years=range(year_min, year_max + 1), observed=True)
    return frozenset(cal.keys())


def local_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (index - pd.Timedelta(hours=1)).tz_convert(config.LOCAL_TZ)


def season_of(index: pd.DatetimeIndex) -> np.ndarray:
    m = local_start(index).month
    return np.select([np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8])],
                     ["winter", "spring", "summer"], "autumn")


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    loc = local_start(index)
    hol = _holiday_dates(loc.year.min() - 1, loc.year.max() + 1)
    dates = pd.Series(loc.date, index=index)
    is_hol = dates.isin(hol).to_numpy()
    prev_hol = (dates - pd.Timedelta(days=1)).isin(hol).to_numpy()
    next_hol = (dates + pd.Timedelta(days=1)).isin(hol).to_numpy()
    hour = loc.hour.to_numpy()
    dow = loc.dayofweek.to_numpy()
    doy = loc.dayofyear.to_numpy()
    out = pd.DataFrame(
        {
            "hour": hour,
            "dow": dow,
            "month": loc.month.to_numpy(),
            "is_weekend": (dow >= 5).astype(int),
            "is_holiday": is_hol.astype(int),
            "is_holiday_adjacent": (prev_hol | next_hol).astype(int),
            "is_offday": ((dow >= 5) | is_hol).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_dst": np.array([bool(t.dst()) for t in loc], dtype=int),
        },
        index=index,
    )
    return out

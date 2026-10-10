"""Shared fixtures: a small synthetic hourly dataset with the processed schema.

Tests never need the network or the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast import config
from gridcast.config import Fold


def synthetic_frame(n_days: int = 500, seed: int = 0, start: str = "2023-01-01 06:00") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days * 24, freq="h", tz="UTC", name="utc_end")
    t = np.arange(len(idx))
    hour = (t % 24)
    dow = (t // 24) % 7
    temp = 20 + 10 * np.sin(2 * np.pi * t / (24 * 365.25)) + 5 * np.sin(2 * np.pi * (hour - 15) / 24) + rng.normal(0, 1.5, len(t))
    load = (40000 * (1 + t / len(t) * 0.2) * (1 + 0.15 * np.sin(2 * np.pi * (hour - 17) / 24))
            * (1 - 0.07 * (dow >= 5)) + 600 * np.clip(temp - 18.3, 0, None) + 300 * np.clip(18.3 - temp, 0, None)
            + rng.normal(0, 400, len(t)))
    df = pd.DataFrame({"demand_raw": load, "demand": load, "is_imputed": False,
                       "eia_day_ahead_forecast": load * (1 + rng.normal(0, 0.02, len(t)))}, index=idx)
    df["temp"] = temp
    df["rh"] = 60 + rng.normal(0, 10, len(t))
    df["wind"] = np.abs(rng.normal(4, 2, len(t)))
    df["tmin"] = temp - 3
    df["tmax"] = temp + 3
    for d in config.FORECAST_LEAD_DAYS:
        noise = rng.normal(0, 0.5 + 0.3 * d, len(t))
        df[f"temp_d{d}"] = temp + noise
        df[f"rh_d{d}"] = df["rh"]
        df[f"wind_d{d}"] = df["wind"]
        df[f"tmin_d{d}"] = df["tmin"] + noise
        df[f"tmax_d{d}"] = df["tmax"] + noise
        for v in ("temp", "rh", "wind", "tmin", "tmax"):
            df[f"{v}_c{d}"] = df[f"{v}_d{d}"]
    return df


@pytest.fixture(scope="session")
def frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="session")
def fold(frame) -> Fold:
    end = frame.index[-1] - pd.Timedelta(days=8)
    test_start = end - pd.Timedelta(days=30)
    cal = test_start - pd.Timedelta(days=30)
    val = cal - pd.Timedelta(days=30)
    return Fold("t1", val.floor("D"), cal.floor("D"), test_start.floor("D"), end.floor("D"),
                train_start=frame.index[0] + pd.Timedelta(days=20))

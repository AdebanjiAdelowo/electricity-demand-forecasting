"""Central study configuration: paths, region, splits, horizons, quantiles.

Everything that defines the forecasting problem lives here so that every
experiment, the API and the copilot read the same definitions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("GRIDCAST_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PREDICTIONS = RESULTS / "predictions"
RUNS = RESULTS / "runs"

# --- region and time ---------------------------------------------------------
REGION = "TEX"  # EIA-930 region; its only balancing authority is ERCOT (ERCO)
REGION_NAME = "ERCOT (Electric Reliability Council of Texas)"
LOCAL_TZ = "America/Chicago"
FREQ = "h"

EIA_URL = "https://www.eia.gov/electricity/gridmonitor/knownissues/xls/Region_TEX.xlsx"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
NWP_MODEL = "gfs_seamless"

# Weather points: the largest ERCOT load centres. Weights are approximate
# shares of 2020 Census metropolitan population among these seven areas and are
# a modelling assumption, not an ERCOT product.
WEATHER_POINTS = {
    "dallas_fort_worth": (32.90, -97.04, 0.360),
    "houston": (29.76, -95.37, 0.335),
    "san_antonio": (29.42, -98.49, 0.120),
    "austin": (30.27, -97.74, 0.107),
    "mcallen": (26.20, -98.23, 0.041),
    "corpus_christi": (27.80, -97.40, 0.020),
    "midland": (31.99, -102.08, 0.017),
}
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
FORECAST_LEAD_DAYS = [1, 2, 3, 4, 5, 6, 7]
NWP_PUBLICATION_DELAY_H = 6  # assumed delay between run initialisation and availability

# --- forecasting problem -----------------------------------------------------
HEADLINE_HORIZONS = [1, 24, 48, 168]
DENSE_HORIZONS = [1, 2, 3, 6, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168]
QUANTILES = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
INTERVAL_ALPHAS = [0.05, 0.1, 0.2, 0.5]  # central intervals with coverage 1 - alpha
WEATHER_REGIMES = ["none", "forecast", "oracle"]
SEASONAL_PERIOD = 168

# --- temporal splits (all timestamps are UTC, hour ending) -------------------
DATA_START = pd.Timestamp("2015-07-01 06:00", tz="UTC")
TRAIN_START = pd.Timestamp("2015-08-01", tz="UTC")
DEV_START = pd.Timestamp("2023-07-01", tz="UTC")
TEST_START = pd.Timestamp("2024-07-01", tz="UTC")
TEST_END = pd.Timestamp("2026-07-01", tz="UTC")  # exclusive
DATA_END = pd.Timestamp("2026-09-01", tz="UTC")  # exclusive; later rows are still being revised


@dataclass(frozen=True)
class Fold:
    """One expanding-window backtest fold.

    fit   : [TRAIN_START, val_start)  parameters are estimated here
    val   : [val_start, cal_start)    early stopping / tuning (deep models)
    cal   : [cal_start, test_start)   conformal calibration only
    test  : [test_start, test_end)    forecast origins that are scored

    Tabular and statistical models, which use fixed hyperparameters, fit on
    fit + val. Nothing is ever fit on cal or test.
    """

    name: str
    val_start: pd.Timestamp
    cal_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_start: pd.Timestamp = TRAIN_START


def make_folds(
    start: pd.Timestamp = TEST_START,
    end: pd.Timestamp = TEST_END,
    months: int = 6,
    cal_days: int = 91,
    val_days: int = 91,
    prefix: str = "test",
) -> list[Fold]:
    folds = []
    s = start
    i = 1
    while s < end:
        e = min(s + pd.DateOffset(months=months), end)
        cal = s - pd.Timedelta(days=cal_days)
        val = cal - pd.Timedelta(days=val_days)
        folds.append(Fold(f"{prefix}{i}", val, cal, s, e))
        s = e
        i += 1
    return folds


TEST_FOLDS = make_folds()
DEV_FOLD = make_folds(DEV_START, TEST_START, months=12, prefix="dev")[0]
# Historical stress fold around Winter Storm Uri (February 2021). Only reanalysis
# weather exists for this period, so only the "none" and "oracle" regimes apply.
URI_FOLD = make_folds(
    pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-04-01", tz="UTC"), months=3, prefix="uri"
)[0]


@dataclass
class StudyConfig:
    seed: int = 0
    horizons: list[int] = field(default_factory=lambda: list(HEADLINE_HORIZONS))
    quantiles: list[float] = field(default_factory=lambda: list(QUANTILES))


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, TABLES, FIGURES, PREDICTIONS, RUNS):
        p.mkdir(parents=True, exist_ok=True)


def forecast_lead_day(h: int) -> int:
    """Legacy NWP lead bucket under an unverified publication-delay assumption.

    Open-Meteo `previous_dayN` holds the value predicted N*24 h before valid
    time. A run needs roughly NWP_PUBLICATION_DELAY_H hours to be published, so
    for horizon h we need N*24 >= h + delay. The archive stops at day 7, so the
    168 h horizon uses day 7 (documented approximation, not delay-safe).
    This function cannot establish issue-time availability. Strict experiments
    use features.available_weather with explicit issue/available timestamps.
    """
    n = -(-(h + NWP_PUBLICATION_DELAY_H) // 24)
    return int(min(max(n, 1), max(FORECAST_LEAD_DAYS)))

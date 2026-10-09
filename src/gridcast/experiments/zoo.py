"""Model registry: name -> factory. One place decides what is compared."""

from __future__ import annotations

from collections.abc import Callable

from gridcast import config
from gridcast.features.build import FeatureSpec
from gridcast.models.base import Forecaster
from gridcast.models.dshw import DoubleSeasonalHW
from gridcast.models.naive import EiaDayAhead, Persistence, SeasonalNaive, SeasonalProfile
from gridcast.models.sarimax import SarimaxForecaster
from gridcast.models.tabular import LightGBMForecaster, RidgeForecaster

W = FeatureSpec(regime="oracle")  # weather-aware spec; the test-time regime is chosen at predict time
N = FeatureSpec(regime="none")

FAMILY_ORDER = ["external", "statistical", "ml", "deep", "foundation"]


def _deep(arch: str, weather: bool, **kw) -> Callable[[], Forecaster]:
    def make():
        from gridcast.models.deep import DeepForecaster, TrainConfig

        return DeepForecaster(TrainConfig(arch=arch, weather=weather, **kw))

    return make


def classical() -> dict[str, Callable[[], Forecaster]]:
    return {
        "persistence": Persistence,
        "seasonal_naive_24h": lambda: SeasonalNaive(24),
        "seasonal_naive_week": lambda: SeasonalNaive(168),
        "seasonal_profile_4w": lambda: SeasonalProfile(4),
        "dshw": DoubleSeasonalHW,
        "sarimax": lambda: SarimaxForecaster(weather=False),
        "sarimax_weather": lambda: SarimaxForecaster(weather=True),
        "ridge": lambda: RidgeForecaster(N, "ridge"),
        "ridge_weather": lambda: RidgeForecaster(W, "ridge_weather"),
        "lightgbm": lambda: LightGBMForecaster(N, "lightgbm", quantile_horizons=config.HEADLINE_HORIZONS),
        "lightgbm_weather": lambda: LightGBMForecaster(W, "lightgbm_weather", quantile_horizons=config.HEADLINE_HORIZONS),
    }


def deep() -> dict[str, Callable[[], Forecaster]]:
    return {
        "gru": _deep("gru", False),
        "gru_weather": _deep("gru", True),
        "tcn_weather": _deep("tcn", True, hidden=64, layers=6),
        "nbeats_weather": _deep("nbeats", True, hidden=256, layers=4),
    }


def external() -> dict[str, Callable[[], Forecaster]]:
    return {"ercot_day_ahead_reference": EiaDayAhead}


def all_models() -> dict[str, Callable[[], Forecaster]]:
    return {**classical(), **deep()}

"""Reuse Ridge fitting and preprocessing with explicit weather configuration."""
from __future__ import annotations

from dataclasses import asdict

from gridcast.features.available_weather import WeatherConfig, features
from gridcast.models.tabular import RidgeForecaster


class WeatherRidge(RidgeForecaster):
    def __init__(self, weather: WeatherConfig, archive=None):
        super().__init__(name=f"ridge_{weather.mode}")
        self.weather = weather
        self.archive = archive
        self.regimes = (weather.mode,)

    def _train_xy(self, df, h):
        return features(df, h, self.weather, self.archive)

    def _xy(self, df, h, regime):
        if regime != self.weather.mode:
            raise ValueError("Model and prediction information regimes differ")
        return self._train_xy(df, h)

    def fit(self, df, fold, horizons):
        super().fit(df, fold, horizons)
        self.info.extra["weather_config"] = asdict(self.weather)

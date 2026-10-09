"""Regression with SARIMA errors on log demand (statsmodels state space).

    log y_t = x_t' beta + u_t,   u_t ~ SARIMA(p,0,q)(P,1,Q)_24

x_t holds weekly Fourier terms and an off-day indicator, plus heating and
cooling degree terms in the weather variant.

Parameters are estimated once per fold on the most recent `fit_days` of the
fitting period. Forecasts for every origin then come from a single Kalman
filter pass with fixed parameters: with filtered state a_{t|t}, transition T
and design Z,

    E[log y_{t+h} | y_1..t] = x_{t+h}' beta + Z T^h a_{t|t}

which is exactly what refiltering up to t and forecasting h steps would give,
at a tiny fraction of the cost. The Gaussian h-step variance gives model-based
(lognormal) predictive quantiles.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from gridcast import config
from gridcast.features.build import BASE_TEMP_C, target_weather
from gridcast.features.calendar import calendar_frame
from gridcast.models.base import Forecaster, fit_slice, long_frame


def _calendar_exog(index: pd.DatetimeIndex, k_week: int = 3) -> pd.DataFrame:
    cal = calendar_frame(index)
    how = cal["dow"].to_numpy() * 24 + cal["hour"].to_numpy()
    x = {"offday_holiday": (cal["is_holiday"]).to_numpy(float)}
    for k in range(1, k_week + 1):
        x[f"wk_sin{k}"] = np.sin(2 * np.pi * k * how / 168)
        x[f"wk_cos{k}"] = np.cos(2 * np.pi * k * how / 168)
    return pd.DataFrame(x, index=index)


def _weather_exog(temp: pd.Series) -> pd.DataFrame:
    cdd = (temp - BASE_TEMP_C).clip(lower=0)
    hdd = (BASE_TEMP_C - temp).clip(lower=0)
    return pd.DataFrame({"cdd": cdd, "cdd2": cdd**2, "hdd": hdd, "hdd2": hdd**2})


class SarimaxForecaster(Forecaster):
    family = "statistical"
    strategy = "recursive"
    probabilistic = True

    def __init__(self, weather: bool = False, order=(2, 0, 1), seasonal_order=(0, 1, 1, 24), fit_days: int = 365,
                 warmup_days: int = 60, maxiter: int = 60):
        super().__init__()
        self.weather = weather
        self.order, self.seasonal_order = order, seasonal_order
        self.fit_days, self.warmup_days, self.maxiter = fit_days, warmup_days, maxiter
        self.name = "sarimax_weather" if weather else "sarimax"
        self.regimes = ("forecast", "oracle") if weather else ("none",)
        self.params = None

    def _exog(self, df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        x = _calendar_exog(index)
        if self.weather:
            x = pd.concat([x, _weather_exog(df["temp"].reindex(index).ffill().bfill())], axis=1)
        return x

    def _model(self, df: pd.DataFrame, index: pd.DatetimeIndex) -> SARIMAX:
        endog = np.log(df["demand"].reindex(index).to_numpy(float))
        return SARIMAX(endog, exog=self._exog(df, index).to_numpy(float), order=self.order,
                       seasonal_order=self.seasonal_order, enforce_stationarity=False,
                       enforce_invertibility=False, concentrate_scale=False)

    def fit(self, df, fold, horizons):
        idx = fit_slice(df, fold).index[-self.fit_days * 24:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._model(df, idx).fit(disp=False, maxiter=self.maxiter, method="lbfgs")
        self.params = np.asarray(res.params)
        self.info.n_params = int(len(self.params))
        self.info.extra = {"aic": float(res.aic), "converged": bool(res.mle_retvals.get("converged", False))}

    def predict(self, df, origins, horizons, regime):
        start = origins[0] - pd.Timedelta(days=self.warmup_days)
        idx = df.index[(df.index >= start) & (df.index <= origins[-1])]
        mod = self._model(df, idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mod.filter(self.params)
        k_exog = mod.k_exog
        beta = self.params[:k_exog]
        Z = res.filter_results.design[:, :, 0]  # (1, m)
        T = res.filter_results.transition[:, :, 0]
        RQR = res.filter_results.selection[:, :, 0] @ res.filter_results.state_cov[:, :, 0] @ res.filter_results.selection[:, :, 0].T
        H = float(res.filter_results.obs_cov[0, 0, 0])
        a = res.filtered_state  # (m, n)
        P = res.filtered_state_cov[:, :, -1]  # steady state after warm-up
        pos = idx.get_indexer(origins)
        zq = norm.ppf(config.QUANTILES)

        frames = []
        hmax = max(horizons)
        Th = np.eye(T.shape[0])
        acc = np.zeros_like(P)
        for h in range(1, hmax + 1):
            acc = T @ acc @ T.T + RQR  # sum_{j<h} T^j RQR T^j'
            Th = T @ Th
            if h not in horizons:
                continue
            target_idx = origins + pd.Timedelta(hours=h)
            x = _calendar_exog(target_idx)
            if self.weather:
                w = target_weather(df, h, regime)["temp"].reindex(target_idx)
                # fall back to the last observed temperature when a forecast value is missing
                w = pd.Series(w.to_numpy(), index=origins).fillna(df["temp"].reindex(origins)).to_numpy()
                x = pd.concat([x, _weather_exog(pd.Series(w, index=target_idx))], axis=1)
            mu = x.to_numpy(float) @ beta + (Z @ Th @ a[:, pos]).ravel()
            var = float((Z @ (Th @ P @ Th.T + acc) @ Z.T)[0, 0]) + H
            q = np.exp(mu[:, None] + np.sqrt(max(var, 0.0)) * zq[None, :])
            frames.append(long_frame(origins, h, regime, np.exp(mu), q))
        return pd.concat(frames, ignore_index=True)

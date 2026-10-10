"""Direct per-horizon tabular forecasters: ridge regression and LightGBM."""

from __future__ import annotations

import os
from dataclasses import replace

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gridcast import config
from gridcast.features.build import FeatureSpec, finite_rows, make_features, one_hot_design
from gridcast.models.base import Forecaster, long_frame, train_mask

CATEGORICAL = ["c_hour", "c_dow"]
N_JOBS = int(os.environ.get("GRIDCAST_JOBS", "6"))  # shared machine: do not take every core


class _Tabular(Forecaster):
    family = "ml"
    strategy = "direct"

    def __init__(self, spec: FeatureSpec | None = None, name: str | None = None):
        super().__init__()
        self.spec = spec or FeatureSpec()
        self.regimes = ("forecast", "oracle") if self.spec.weather else ("none",)
        if name:
            self.name = name
        self.models: dict = {}
        self.feature_names: dict[int, list[str]] = {}

    def _xy(self, df, h, regime):
        return make_features(df, h, replace(self.spec, regime=regime) if self.spec.weather else self.spec)

    def _train_xy(self, df, h):
        return make_features(df, h, self.spec.training_view())


class RidgeForecaster(_Tabular):
    name = "ridge"

    def __init__(self, spec=None, name=None, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
        super().__init__(spec, name)
        self.alphas = alphas
        self.medians: dict[int, pd.Series] = {}

    def fit(self, df, fold, horizons):
        for h in horizons:
            X, y, _ = self._train_xy(df, h)
            X = one_hot_design(X)
            ok = finite_rows(X, y)
            tr = train_mask(X.index, h, fold.train_start, fold.val_start) & ok
            va = train_mask(X.index, h, fold.val_start, fold.cal_start) & ok
            best, best_err = None, np.inf
            for a in self.alphas:  # temporal hold-out for the penalty, then refit on fit + val
                m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr], y[tr])
                err = np.mean(np.abs(m.predict(X[va]) - y[va]))
                if err < best_err:
                    best, best_err = a, err
            full = tr | va
            self.models[h] = make_pipeline(StandardScaler(), Ridge(alpha=best)).fit(X[full], y[full])
            self.medians[h] = X[full].median()
            self.feature_names[h] = list(X.columns)
            self.info.extra[f"alpha_h{h}"] = best
        self.info.n_params = int(sum(len(v) + 1 for v in self.feature_names.values()))

    def predict(self, df, origins, horizons, regime):
        frames = []
        for h in horizons:
            X, _, m = self._xy(df, h, regime)
            X = one_hot_design(X).reindex(origins)
            X = X.replace([np.inf, -np.inf], np.nan).fillna(self.medians[h])
            frames.append(long_frame(origins, h, regime, self.models[h].predict(X) * m.reindex(origins).to_numpy()))
        return pd.concat(frames, ignore_index=True)


class LightGBMForecaster(_Tabular):
    name = "lightgbm"
    probabilistic = True

    def __init__(self, spec=None, name=None, quantiles: bool = True, n_estimators: int = 600, num_leaves: int = 63,
                 learning_rate: float = 0.05, q_estimators: int = 250, q_leaves: int = 31, seed: int = 0,
                 quantile_horizons: list[int] | None = None):
        super().__init__(spec, name)
        self.quantiles = quantiles
        self.quantile_horizons = quantile_horizons  # None = all horizons
        self.probabilistic = quantiles
        self.p = dict(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
                      min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=1.0, random_state=seed, n_jobs=N_JOBS, verbose=-1)
        self.qp = dict(self.p, n_estimators=q_estimators, num_leaves=q_leaves, learning_rate=0.08)
        self.qmodels: dict = {}

    def fit(self, df, fold, horizons):
        n_nodes = 0
        for h in horizons:
            X, y, _ = self._train_xy(df, h)
            tr = train_mask(X.index, h, fold.train_start, fold.cal_start) & np.isfinite(y.to_numpy(float))
            cats = [c for c in CATEGORICAL if c in X.columns]
            Xt, yt = X[tr], y[tr]
            self.models[h] = lgb.LGBMRegressor(objective="l2", **self.p).fit(Xt, yt, categorical_feature=cats)
            self.feature_names[h] = list(X.columns)
            n_nodes += self.p["n_estimators"] * self.p["num_leaves"]
            if self._has_q(h):
                for q in config.QUANTILES:
                    self.qmodels[(h, q)] = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.qp).fit(
                        Xt, yt, categorical_feature=cats)
        self.info.n_params = int(n_nodes)

    def predict(self, df, origins, horizons, regime):
        frames = []
        for h in horizons:
            X, _, m = self._xy(df, h, regime)
            X = X.reindex(origins)
            mm = m.reindex(origins).to_numpy()
            yhat = self.models[h].predict(X) * mm
            q = None
            if self._has_q(h):
                q = np.column_stack([self.qmodels[(h, lv)].predict(X) for lv in config.QUANTILES]) * mm[:, None]
            frames.append(long_frame(origins, h, regime, yhat, q))
        return pd.concat(frames, ignore_index=True)

    def _has_q(self, h: int) -> bool:
        return self.quantiles and (self.quantile_horizons is None or h in self.quantile_horizons)

    def importance(self, h: int, kind: str = "gain") -> pd.Series:
        b = self.models[h].booster_
        s = pd.Series(b.feature_importance(importance_type=kind), index=b.feature_name(), dtype=float)
        return (s / s.sum()).sort_values(ascending=False)

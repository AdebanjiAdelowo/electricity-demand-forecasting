"""Fidelity, feasibility, diversity and held-out simulated forecasting utility."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .network import solve


def acf(x: np.ndarray, max_lag: int = 12) -> np.ndarray:
    a = x - x.mean(axis=1, keepdims=True)
    variance = np.mean(a * a, axis=1)
    vals = []
    for lag in range(1, max_lag + 1):
        cov = np.mean(a[:, :-lag] * a[:, lag:], axis=1)
        vals.append(np.mean(cov / np.maximum(variance, 1e-12), axis=0))
    return np.stack(vals)


def evaluate(x: np.ndarray, heldout: np.ndarray) -> dict:
    state = solve(x)
    a, b = x.reshape(-1, 5), heldout.reshape(-1, 5)
    total, reference_total = x[:, :, :4].sum(axis=-1), heldout[:, :, :4].sum(axis=-1)
    distances = pdist(x.reshape(len(x), -1)) / np.sqrt(np.prod(x.shape[1:]))
    return {
        "mean_absolute_difference_mw": float(np.mean(abs(a.mean(axis=0) - b.mean(axis=0)))),
        "variance_absolute_difference_mw2": float(np.mean(abs(a.var(axis=0) - b.var(axis=0)))),
        "marginal_wasserstein_mw": float(np.mean([wasserstein_distance(a[:, j], b[:, j]) for j in range(5)])),
        "acf_absolute_difference": float(np.mean(abs(acf(x) - acf(heldout)))),
        "cross_correlation_absolute_difference": float(np.mean(abs(np.corrcoef(a.T) - np.corrcoef(b.T)))),
        "peak_wasserstein_mw": float(wasserstein_distance(total.max(axis=1), reference_total.max(axis=1))),
        "diversity_pairwise_rms_mw": float(distances.mean()),
        "near_duplicate_pair_fraction": float(np.mean(distances < 1e-3)),
        "mean_hourly_change_mw": float(np.mean(abs(np.diff(x, axis=1)))),
        "power_flow_convergence_rate": float(state["converged"].mean()),
        "voltage_violation_rate": float(state["voltage_bad"].mean()),
        "line_overload_rate": float(state["thermal_bad"].mean()),
        "bounds_violation_rate": float(state["bounds_bad"].mean()),
        "slack_limit_violation_rate": float(state["slack_bad"].mean()),
        "hour_feasible_fraction": float(state["feasible"].mean()),
        "trajectory_feasible_fraction": float(state["feasible"].all(axis=1).mean()),
        "max_nodal_residual_mva": float(state["residual"].max()),
    }


def forecasting_rows(windows: np.ndarray):
    y = windows[:, :, :4].sum(axis=-1)
    h = np.arange(3, y.shape[1])
    features = [y[:, h - lag].reshape(-1) for lag in (1, 2, 3)]
    features += [np.tile(np.sin(2 * np.pi * h / 24), len(y)), np.tile(np.cos(2 * np.pi * h / 24), len(y))]
    return np.column_stack(features), y[:, h].reshape(-1)


def utility(train: np.ndarray, heldout: np.ndarray) -> dict:
    x, y = forecasting_rows(train)
    xt, yt = forecasting_rows(heldout)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(x, y)
    forecast = model.predict(xt)
    return {"utility_mae_mw": float(np.mean(abs(yt - forecast))),
            "utility_rmse_mw": float(np.sqrt(np.mean((yt - forecast) ** 2))),
            "utility_n_test_hours": len(yt)}

"""Deterministic CPU checks of physical equations and evaluation separation."""
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from gridcast import config
from gridcast.models.base import train_mask
from gridcast.models.tabular import RidgeForecaster

# Standalone experiments are intentionally outside the installed gridcast package.
sys.path.insert(0, str(config.ROOT))
from experiments.physics_informed_synthetic_energy.evaluation import evaluate, forecasting_rows
from experiments.physics_informed_synthetic_energy.generator import GaussianWindows, reference
from experiments.physics_informed_synthetic_energy.network import ANCHOR, Z, pandapower_check, project, solve


def test_ac_zero_load_and_two_bus_analytic():
    empty = solve(np.zeros(5))
    np.testing.assert_allclose(empty["voltage"], 1)
    np.testing.assert_allclose(empty["line_current"], 0)
    x = np.array([0.2, 0, 0, 0, 0])
    state = solve(x)
    v = state["voltage"][1]
    current = (1 - v) / Z[0]
    assert v * np.conj(current) == pytest.approx(0.2 + 0.05j, abs=1e-9)
    assert state["slack_p"] > 0.2  # losses are positive
    assert state["residual"] < 1e-8


def test_convergence_is_distinct_from_operating_limits():
    state = solve(np.array([0.4, 0.3, 0.3, 0.3, 0.0]))
    assert state["converged"]
    assert state["voltage_bad"] or state["thermal_bad"]
    assert not state["feasible"]
    negative = solve(np.array([-0.01, 0, 0, 0, 0]))
    assert negative["bounds_bad"] and not negative["feasible"]


def test_failure_handling():
    assert not solve(ANCHOR, max_iter=1)["converged"]
    assert not solve(ANCHOR, max_iter=1)["feasible"]
    with pytest.raises(ValueError):
        solve(np.array([np.nan] * 5))
    with pytest.raises(ValueError):
        project(np.zeros((2, 4)))


def test_projection_preserves_feasible_and_corrects_bad_windows():
    a = np.tile(ANCHOR, (2, 24, 1))
    a[1, :, :4] = 0.8
    a[1, :, 4] = -0.1
    corrected, alpha = project(a)
    np.testing.assert_allclose(corrected[0], a[0])
    assert alpha[0] == 1 and alpha[1] < 1
    assert solve(corrected)["feasible"].all()


def test_generator_reproducibility_shapes_and_reference_separation():
    train, _ = reference(16, 1)
    heldout, _ = reference(8, 2)
    model = GaussianWindows().fit(train, rank=4)
    a = model.sample(8, 4)
    assert a.shape == (8, 24, 5)
    np.testing.assert_array_equal(a, model.sample(8, 4))
    assert not np.array_equal(a, model.sample(8, 5))
    np.testing.assert_array_equal(train, reference(16, 1)[0])
    assert not np.array_equal(train[:8], heldout)
    assert solve(train)["feasible"].all()


def test_fidelity_identity_and_utility_alignment():
    a, _ = reference(8, 2)
    result = evaluate(a, a)
    assert result["marginal_wasserstein_mw"] == 0
    assert result["acf_absolute_difference"] == 0
    assert result["trajectory_feasible_fraction"] == 1
    x, y = forecasting_rows(a)
    assert x[0, 0] == pytest.approx(a[0, 2, :4].sum())
    assert y[0] == pytest.approx(a[0, 3, :4].sum())
    changed = a.copy()
    changed[0, 3:, :4] += 1
    np.testing.assert_array_equal(x[0], forecasting_rows(changed)[0][0])


def test_target_boundary_and_splits():
    for fold in config.TEST_FOLDS:
        assert fold.train_start < fold.val_start < fold.cal_start < fold.test_start < fold.test_end
        idx = pd.date_range(fold.cal_start - pd.Timedelta(hours=48), fold.cal_start, freq="h")
        mask = train_mask(idx, 24, fold.train_start, fold.cal_start)
        assert (idx[mask] + pd.Timedelta(hours=24) < fold.cal_start).all()


def test_fitted_scaler_and_regression_ignore_post_training_data(frame, fold):
    original = RidgeForecaster(alphas=(1.0,))
    original.fit(frame, fold, [24])
    tampered = frame.copy()
    tampered.loc[tampered.index >= fold.cal_start, "demand"] *= 9
    altered = RidgeForecaster(alphas=(1.0,))
    altered.fit(tampered, fold, [24])
    for name, attribute in [("standardscaler", "mean_"), ("standardscaler", "scale_"), ("ridge", "coef_")]:
        np.testing.assert_array_equal(getattr(original.models[24].named_steps[name], attribute),
                                      getattr(altered.models[24].named_steps[name], attribute))


@pytest.mark.skipif(importlib.util.find_spec("pandapower") is None, reason="Optional physics dependency")
def test_independent_pandapower_solver():
    a, _ = reference(2, 6)
    points = np.vstack([np.zeros(5), ANCHOR, a[:, ::6].reshape(-1, 5)])
    result = pandapower_check(points)
    assert result["failures"] == 0
    assert result["max_voltage_error_pu"] < 1e-7

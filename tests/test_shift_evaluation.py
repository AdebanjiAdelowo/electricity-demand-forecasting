import numpy as np
import pandas as pd
import pytest

from gridcast.evaluation.shift import residual_drift, slices
from gridcast.experiments.weather_uncertainty import interval_table


def test_shift_diagnostic_detects_known_offset_without_pvalue():
    dates = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    p = pd.DataFrame({"model": "m", "regime": "none", "h": 1, "fold": ["f1"] * 24 + ["f2"] * 24,
                      "yhat": 10, "y": [10] * 24 + [12] * 24, "target_time": dates})
    result = residual_drift(p, {"f1": 1, "f2": 1}).iloc[0]
    assert result.residual_wasserstein_mw == 2
    assert result.ks_statistic_descriptive == 1
    assert not any("pvalue" in c for c in result.index)


def test_slices_use_local_interval_start_and_fixed_training_thresholds():
    p = pd.DataFrame({"target_time": pd.to_datetime(["2024-07-06 05:00Z", "2024-07-06 12:00Z"]),
                      "fold": "f1", "y": [15, 100]})
    out = slices(p, {"f1": [10, 20, 90]})
    assert out.week_part.tolist() == ["weekday", "weekend"]
    assert out.day_night.tolist() == ["night_18_06", "day_06_18"]
    assert out.demand_regime.tolist() == ["mid", "peak_top5pct"]


def test_winkler_score_known_hit_and_miss():
    p = pd.DataFrame({"model": "m", "alpha": 0.1, "y": [0, 2], "lo": [-1, -1], "hi": [1, 1]})
    score = interval_table(p, ["model"]).iloc[0]
    assert score.coverage == 0.5
    assert score.width_mw == 2
    assert score.winkler_mw == pytest.approx(np.mean([2, 22]))

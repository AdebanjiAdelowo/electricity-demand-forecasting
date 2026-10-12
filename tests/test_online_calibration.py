import numpy as np
import pandas as pd
import pytest

from gridcast.probabilistic.online import OnlineConfig, calibrate_online


def predictions(h=24):
    # Deliberately irregular forecast origins with a six-hour gap.
    origins = pd.date_range("2024-01-01", periods=360, freq="h", tz="UTC").delete(np.arange(45, 51))
    i = np.arange(len(origins))
    return pd.DataFrame({"model": "test", "regime": "none", "fold": "f1", "h": h,
                         "origin": origins, "target_time": origins + pd.Timedelta(hours=h),
                         "yhat": 100.0, "y": 100 + (i % 17), "target_imputed": False,
                         "block": np.where(i < 75, "cal", "test")})


@pytest.mark.parametrize("mode", ["rolling", "adaptive"])
@pytest.mark.parametrize("h", [1, 24, 48, 168])
def test_timestamp_gate_with_irregular_origins_and_delayed_feedback(mode, h):
    p = predictions(h)
    cfg = OnlineConfig(mode=mode, window=30, min_scores=5, feedback_delay_hours=3)
    a = calibrate_online(p, cfg, (0.1,))
    cut = a.origin.iloc[15]
    changed = p.copy()
    changed.loc[changed.target_time + pd.Timedelta(hours=3) > cut, "y"] += 1e6
    b = calibrate_online(changed, cfg, (0.1,))
    keep = a.origin <= cut
    np.testing.assert_allclose(a.loc[keep, "lo"], b.loc[keep, "lo"], equal_nan=True)
    np.testing.assert_allclose(a.loc[keep, "effective_alpha"], b.loc[keep, "effective_alpha"])
    known = a.last_feedback_at.notna()
    assert (a.loc[known, "last_feedback_at"] <= a.loc[known, "origin"]).all()
    assert (a.n_scores <= 30).all()
    assert (a.feedback_at == a.target_time + pd.Timedelta(hours=3)).all()


def test_quantile_uses_actual_finite_count_and_imputed_exclusion():
    p = predictions(1)
    p.loc[:60, "target_imputed"] = True
    p.loc[61:65, "y"] = np.nan
    a = calibrate_online(p, OnlineConfig(window=100, min_scores=1), (0.2,))
    row = a.iloc[0]
    eligible = p[(p.target_time <= row.origin) & ~p.target_imputed & p.y.notna()]
    scores = sorted(abs(eligible.y - eligible.yhat))
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * 0.8)))
    assert row.n_scores == len(scores)
    assert row.hi - row.yhat == scores[rank - 1]


def test_adaptive_update_only_after_issued_interval_feedback():
    p = predictions(24)
    p.loc[p.block == "test", "y"] = 10000
    cfg = OnlineConfig(mode="adaptive", window=50, min_scores=5, gamma=0.01)
    a = calibrate_online(p, cfg, (0.1,))
    first_feedback = a.feedback_at.min()
    assert (a.loc[a.origin < first_feedback, "effective_alpha"] == 0.1).all()
    assert a.loc[a.origin == first_feedback, "effective_alpha"].iloc[0] == pytest.approx(0.091)


def test_reproduction_and_bad_alignment():
    p = predictions()
    cfg = OnlineConfig(window=30, min_scores=5)
    pd.testing.assert_frame_equal(calibrate_online(p, cfg), calibrate_online(p, cfg))
    p.loc[0, "target_time"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        calibrate_online(p, cfg)


@pytest.mark.parametrize("kw", [{"mode": "bad"}, {"window": 10, "min_scores": 20},
                               {"feedback_delay_hours": -1}, {"gamma": 0}])
def test_online_config_rejects_invalid(kw):
    with pytest.raises(ValueError):
        OnlineConfig(**kw)

import numpy as np
import pandas as pd
import pytest

from gridcast.experiments.interval_validation import DEV, choose
from gridcast.probabilistic.online import OnlineConfig, calibrate_online


@pytest.mark.parametrize("h", [1, 24, 48, 168])
def test_elapsed_window_causal_scores_gaps_and_delay(h):
    origins = pd.date_range("2023-01-01", periods=700, freq="h", tz="UTC").delete(range(100, 140))
    p = pd.DataFrame(
        dict(
            model="fixture",
            regime="none",
            fold="dev",
            h=h,
            origin=origins,
            target_time=origins + pd.Timedelta(hours=h),
            y=np.arange(len(origins)),
            yhat=0.0,
            target_imputed=False,
            block=np.where(np.arange(len(origins)) < 400, "cal", "test"),
        )
    )
    cfg = OnlineConfig(window=168, window_hours=168, min_scores=168, feedback_delay_hours=3)
    a = calibrate_online(p, cfg, (0.1,))
    for row in a.itertuples():
        arrival = p.target_time + pd.Timedelta(hours=3)
        scores = p[(arrival <= row.origin) & (arrival > row.origin - pd.Timedelta(hours=168))].y.to_numpy()
        assert row.n_scores == len(scores)
        if len(scores) == 168:
            assert row.hi == np.sort(scores)[min(len(scores), int(np.ceil((len(scores) + 1) * 0.9))) - 1]
        else:
            assert np.isnan(row.hi)
    cutoff = a.origin.iloc[10]
    changed = p.copy()
    changed.loc[p.target_time + pd.Timedelta(hours=3) > cutoff, "y"] += 1e9
    b = calibrate_online(changed, cfg, (0.1,))
    np.testing.assert_allclose(
        a.loc[a.origin <= cutoff, "hi"], b.loc[b.origin <= cutoff, "hi"], equal_nan=True
    )


def test_selection_fails_closed_and_uses_winkler_then_shorter_tie():
    p = pd.DataFrame(
        dict(
            model=["ridge"] * 4,
            h=[168] * 4,
            alpha=[0.1] * 4,
            days=[7, 14, 28, 60],
            coverage=[0.9, 0.87, 0.89, 0.91],
            winkler_mw=[1, 2, 3, 3],
            complete=[False, True, True, True],
        )
    )
    assert choose(p)[0]["days"] == 28
    p["coverage"] = 0.88
    assert choose(p)[0]["days"] == 14
    p["coverage"] = 0.85
    assert choose(p)[0]["status"] == "no_candidate_qualified"
    assert choose(p)[0]["days"] is None


def test_disjoint_development_boundaries_and_long_horizon_warmup():
    assert DEV.train_start < DEV.val_start < DEV.cal_start < DEV.test_start < DEV.test_end
    assert DEV.test_end == pd.Timestamp("2024-04-01", tz="UTC")
    assert DEV.test_start - DEV.cal_start >= pd.Timedelta(days=67)


def test_elapsed_complete_hourly_matches_legacy_score_count_window():
    origins = pd.date_range("2023-01-01", periods=600, freq="h", tz="UTC")
    p = pd.DataFrame(
        dict(
            model="fixture",
            regime="none",
            fold="f",
            h=168,
            origin=origins,
            target_time=origins + pd.Timedelta(hours=168),
            y=np.arange(600) % 19,
            yhat=0.0,
            target_imputed=False,
            block=np.where(np.arange(600) < 400, "cal", "test"),
        )
    )
    legacy = calibrate_online(p, OnlineConfig(window=168, min_scores=168), (0.1,))
    elapsed = calibrate_online(p, OnlineConfig(window=168, window_hours=168, min_scores=168), (0.1,))
    pd.testing.assert_frame_equal(legacy, elapsed)


def test_sparse_scores_expire_instead_of_extending_elapsed_window():
    origins = pd.date_range("2023-01-01", periods=400, freq="2h", tz="UTC")
    p = pd.DataFrame(
        dict(
            model="fixture",
            regime="none",
            fold="f",
            h=24,
            origin=origins,
            target_time=origins + pd.Timedelta(hours=24),
            y=1.0,
            yhat=0.0,
            target_imputed=False,
            block=np.where(np.arange(400) < 200, "cal", "test"),
        )
    )
    elapsed = calibrate_online(p, OnlineConfig(window=168, window_hours=168, min_scores=168), (0.1,))
    assert (elapsed.n_scores == 84).all()
    assert elapsed.lo.isna().all()


def test_memory_guard_enforces_approved_limit(monkeypatch):
    from types import SimpleNamespace

    from gridcast.experiments import interval_validation as experiment

    monkeypatch.setattr(experiment, "ACTIVE_MEMORY_LIMIT", 1024)
    monkeypatch.setattr(experiment.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1025))
    with pytest.raises(RuntimeError, match="budget"):
        experiment.guard()
    monkeypatch.setattr(experiment, "ACTIVE_MEMORY_LIMIT", 2048)
    assert experiment.guard() == 1025


def test_periodic_budget_check_checks_time_and_memory(monkeypatch):
    from gridcast.experiments import interval_validation as experiment

    handler = []
    timer = []
    ticks = iter([0.0, 1201.0])
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(experiment, "guard", lambda: 100)
    monkeypatch.setattr(experiment.signal, "signal", lambda _, fn: handler.append(fn))
    monkeypatch.setattr(experiment.signal, "setitimer", lambda *args: timer.append(args))
    experiment.bounded()
    assert timer[0][1:] == (1.0, 1.0)
    with pytest.raises(TimeoutError, match="20-minute"):
        handler[0]()


def test_evaluation_refuses_overwriting_development_directory(tmp_path):
    from gridcast.experiments.interval_validation import evaluate

    with pytest.raises(ValueError, match="new evaluation"):
        evaluate(tmp_path, tmp_path)


def test_paired_bootstrap_is_deterministic_on_matched_keys():
    from gridcast.experiments.interval_validation import paired

    origins = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    static = pd.DataFrame(
        dict(
            model="ridge",
            h=24,
            alpha=0.1,
            fold="f",
            origin=origins,
            target_time=origins + pd.Timedelta(hours=24),
            y=0.0,
            lo=-2.0,
            hi=2.0,
            method="conformal_static",
        )
    )
    selected = static.assign(lo=-1.0, hi=1.0, method="selected_policy")
    frame = pd.concat([static, selected], ignore_index=True)
    first = paired(frame)
    pd.testing.assert_frame_equal(first, paired(frame))
    assert (first.winkler_difference_mw == -2.0).all()
    assert (first.winkler_lo == -2.0).all()
    assert (first.winkler_hi == -2.0).all()
    assert (first.coverage_difference == 0.0).all()
    assert set(first.block_days) == {7, 14, 28}


def test_online_config_preserves_positional_constructor():
    cfg = OnlineConfig("adaptive", 30, 5, 3, 0.02, 0.01, 0.4)
    assert cfg.gamma == 0.02
    assert cfg.alpha_min == 0.01
    assert cfg.alpha_max == 0.4
    assert cfg.window_hours is None

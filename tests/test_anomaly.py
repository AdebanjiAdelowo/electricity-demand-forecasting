import numpy as np
import pandas as pd

from gridcast.anomaly import detectors as det
from gridcast.anomaly.inject import inject
from gridcast.experiments.anomaly_study import event_recall


def _window(frame):
    return frame.index[6000], frame.index[11000]


def test_injection_is_labelled_and_leaves_other_hours_untouched(frame):
    a, b = _window(frame)
    inj = inject(frame, a, b, "dip_block", 0.2, n_events=10, seed=1)
    changed = ~np.isclose(inj.df["demand"], frame["demand"])
    assert (changed == inj.labels.to_numpy()).all()
    assert len(inj.events) == 10 and inj.labels.sum() == inj.events["hours"].sum()
    assert inj.labels[frame.index < a].sum() == 0 and inj.labels[frame.index >= b].sum() == 0
    e = inj.events.iloc[0]
    ratio = (
        inj.df["demand"].iloc[e.start_pos : e.end_pos + 1] / frame["demand"].iloc[e.start_pos : e.end_pos + 1]
    )
    np.testing.assert_allclose(ratio, 0.8)


def test_flatline_injection_and_detector(frame):
    a, b = _window(frame)
    inj = inject(frame, a, b, "flatline", 0.0, n_events=5, seed=2)
    score = det.flatline_score(inj.df["demand"])
    for e in inj.events.itertuples():
        assert score.iloc[e.end_pos] >= min(e.hours, 24) - 1
    assert det.flatline_score(frame["demand"]).max() == 0


def test_robust_seasonal_detector_finds_large_spikes_and_is_causal(frame):
    a, b = _window(frame)
    inj = inject(frame, a, b, "spike", 0.3, n_events=20, seed=3)
    clean = det.score_robust_seasonal(frame)
    dirty = det.score_robust_seasonal(inj.df)
    thr = clean.quantile(0.995)
    rec, delay = event_recall((dirty > thr).to_numpy(), inj.events)
    assert rec >= 0.9 and delay <= 1
    # scores before the first injected hour are identical: nothing looks ahead
    first = inj.events.start_pos.min()
    pd.testing.assert_series_equal(clean.iloc[:first], dirty.iloc[:first])


def test_cusum_accumulates_small_persistent_shifts():
    z = pd.Series(np.r_[np.zeros(200), np.full(100, 1.2), np.zeros(200)])
    s = det.cusum(z)
    assert s.iloc[:200].max() == 0 and s.iloc[299] > 30 and s.iloc[-1] < s.iloc[299]
    assert det.cusum(-z).iloc[299] == s.iloc[299]

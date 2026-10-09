"""The central validity property: nothing after the forecast origin may influence its features."""

import numpy as np
import pandas as pd
import pytest

from gridcast import config
from gridcast.features.build import FeatureSpec, make_features, target_frame
from gridcast.features.calendar import calendar_frame


@pytest.mark.parametrize("h", [1, 24, 48, 168])
@pytest.mark.parametrize("regime", ["none", "forecast"])
def test_future_demand_and_observed_weather_do_not_leak(frame, h, regime):
    spec = FeatureSpec(regime=regime)
    pos = 6000
    origin = frame.index[pos]
    X0, _, m0 = make_features(frame, h, spec)
    tampered = frame.copy()
    after = tampered.index > origin
    tampered.loc[after, "demand"] *= 3.0
    for c in ("temp", "rh", "wind", "tmin", "tmax"):  # observed weather after the origin is unknown
        tampered.loc[after, c] += 25.0
    X1, _, m1 = make_features(tampered, h, spec)
    pd.testing.assert_series_equal(X0.loc[origin], X1.loc[origin])
    assert m0.loc[origin] == m1.loc[origin]
    # and every earlier origin is untouched as well
    np.testing.assert_allclose(X0.iloc[pos - 500:pos + 1].to_numpy(float), X1.iloc[pos - 500:pos + 1].to_numpy(float), equal_nan=True)


def test_oracle_regime_does_use_future_weather(frame):
    """Sanity check of the test above: the oracle regime must react to future observed weather."""
    spec = FeatureSpec(regime="oracle")
    origin = frame.index[6000]
    X0, _, _ = make_features(frame, 24, spec)
    tampered = frame.copy()
    tampered.loc[tampered.index > origin, "temp"] += 25.0
    X1, _, _ = make_features(tampered, 24, spec)
    assert X1.loc[origin, "w_temp"] == pytest.approx(X0.loc[origin, "w_temp"] + 25.0)


def test_forecast_regime_uses_published_lead_bucket(frame):
    for h in (1, 24, 48, 168):
        d = config.forecast_lead_day(h)
        assert d * 24 >= min(h + config.NWP_PUBLICATION_DELAY_H, 7 * 24)
        X, _, _ = make_features(frame, h, FeatureSpec(regime="forecast"))
        origin = frame.index[6000]
        assert X.loc[origin, "w_temp"] == pytest.approx(frame[f"temp_d{d}"].loc[origin + pd.Timedelta(hours=h)])


def test_target_alignment(frame):
    h = 24
    _, y, m = make_features(frame, h, FeatureSpec())
    origin = frame.index[5000]
    assert y.loc[origin] * m.loc[origin] == pytest.approx(frame["demand"].loc[origin + pd.Timedelta(hours=h)])
    assert m.loc[origin] == pytest.approx(frame["demand"].loc[:origin].iloc[-168:].mean())
    tf = target_frame(frame, [h]).set_index("origin")
    assert tf.loc[origin, "y"] == pytest.approx(frame["demand"].loc[origin + pd.Timedelta(hours=h)])


def test_same_hour_lags_point_at_observed_history(frame):
    for h in (1, 24, 30, 48, 168):
        X, _, m = make_features(frame, h, FeatureSpec())
        origin = frame.index[5000]
        target = origin + pd.Timedelta(hours=h)
        j = -(-h // 24)
        ref = target - pd.Timedelta(hours=24 * j)
        assert ref <= origin
        assert X.loc[origin, "l_same_hour_day"] * m.loc[origin] == pytest.approx(frame["demand"].loc[ref])


def test_calendar_handles_dst_and_hour_ending_labels():
    # 2024-03-10 08:00 UTC is 02:00 CST -> the hour ending 03:00 CDT does not start at 02:00 local
    idx = pd.date_range("2024-03-10 05:00", "2024-03-10 12:00", freq="h", tz="UTC")
    cal = calendar_frame(idx)
    assert list(cal["hour"]) == [22, 23, 0, 1, 3, 4, 5, 6]  # local hour 2 never occurs
    idx = pd.date_range("2024-11-03 05:00", "2024-11-03 10:00", freq="h", tz="UTC")
    assert list(calendar_frame(idx)["hour"]) == [23, 0, 1, 1, 2, 3]  # local hour 1 occurs twice
    # the hour ending at local midnight belongs to the previous day
    cal = calendar_frame(pd.DatetimeIndex(["2024-07-05 05:00"], tz="UTC"))  # 00:00 CDT on 5 July
    assert cal["hour"].iloc[0] == 23 and cal["is_holiday"].iloc[0] == 1

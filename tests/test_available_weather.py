import numpy as np
import pandas as pd
import pytest

from gridcast.features.available_weather import WeatherConfig, features, select_issued
from gridcast.features.build import FeatureSpec, make_features
from gridcast.models.available_weather import WeatherRidge


@pytest.mark.parametrize("h", [1, 24, 48, 168])
def test_history_weather_ignores_future_and_delayed_observations(frame, h):
    cfg = WeatherConfig(observation_delay_hours=2)
    cut = frame.index[4000]
    a = features(frame, h, cfg)[0]
    changed = frame.copy()
    changed.loc[changed.index > cut - pd.Timedelta(hours=2), ["temp", "rh", "wind"]] += 10000
    b = features(changed, h, cfg)[0]
    pd.testing.assert_series_equal(a.loc[cut], b.loc[cut])
    assert a.loc[cut, "wx_temp_lag0"] == frame.temp.loc[cut - pd.Timedelta(hours=2)]


def test_none_mode_matches_validated_features(frame):
    a = features(frame, 24, WeatherConfig(mode="none"))
    b = make_features(frame, 24, FeatureSpec())
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_series_equal(a[1], b[1])


def test_missing_weather_is_only_filled_from_past(frame):
    df = frame.copy()
    df.iloc[4000:4008, df.columns.get_loc("temp")] = np.nan
    cfg = WeatherConfig(observation_delay_hours=0, fill_limit_hours=2)
    x = features(df, 24, cfg)[0]
    assert x.wx_temp_lag0.iloc[4001] == df.temp.iloc[3999]
    assert np.isnan(x.wx_temp_lag0.iloc[4002])
    assert x.wx_temp_missing.iloc[4001] == 1


def archive():
    return pd.DataFrame({
        "issue_time": pd.to_datetime(["2024-01-01 00:00Z", "2024-01-01 06:00Z"]),
        "available_at": pd.to_datetime(["2024-01-01 02:00Z", "2024-01-01 08:00Z"]),
        "valid_time": pd.to_datetime(["2024-01-02 00:00Z"] * 2),
        "source": ["synthetic test run"] * 2, "temp": [10.0, 20.0]})


def test_issue_and_publication_time_gate():
    origins = pd.to_datetime(["2024-01-01 01:00Z", "2024-01-01 07:00Z", "2024-01-01 09:00Z"])
    valid = pd.to_datetime(["2024-01-02 00:00Z"] * 3)
    selected = select_issued(archive(), origins, valid, ("temp",))
    assert np.isnan(selected.temp.iloc[0])
    np.testing.assert_array_equal(selected.temp.iloc[1:], [10, 20])
    changed = archive()
    changed.loc[1, "temp"] = 99999
    altered = select_issued(changed, origins, valid, ("temp",))
    assert altered.temp.iloc[1] == selected.temp.iloc[1]
    assert (selected.dropna().available_at <= selected.dropna().origin).all()


@pytest.mark.parametrize("fault", ["naive", "ordering", "missing", "duplicate", "source"])
def test_invalid_archive_rejected(fault):
    df = archive()
    if fault == "naive":
        df["available_at"] = df.available_at.dt.tz_localize(None)
    elif fault == "ordering":
        df.loc[0, "available_at"] = df.loc[0, "issue_time"] - pd.Timedelta(hours=1)
    elif fault == "missing":
        df = df.drop(columns="issue_time")
    elif fault == "duplicate":
        df = pd.concat([df, df.iloc[:1]])
    else:
        df.loc[0, "source"] = ""
    with pytest.raises(ValueError):
        select_issued(df, pd.to_datetime(["2024-01-01 09:00Z"]), pd.to_datetime(["2024-01-02 00:00Z"]), ("temp",))


@pytest.mark.parametrize("kwargs", [{"mode": "forecast"}, {"observation_delay_hours": -1},
                                    {"lags": (-1,)}, {"rolling_hours": (0,)}, {"variables": ("unknown",)}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        WeatherConfig(**kwargs)


def test_issued_mode_cannot_accept_legacy_bucket_data(frame):
    with pytest.raises(ValueError, match="timestamped archive"):
        features(frame, 24, WeatherConfig(mode="issued"))


def test_oracle_is_labelled_and_responds_to_future(frame):
    cfg = WeatherConfig(mode="oracle")
    x = features(frame, 24, cfg)[0]
    assert x.attrs["information_mode"] == "oracle"
    assert not x.attrs["operational_availability_verified"]
    assert x.wx_temp_target_lag0.iloc[4000] == frame.temp.iloc[4024]


def test_weather_ridge_smoke_and_regime_validation(frame, fold):
    model = WeatherRidge(WeatherConfig(rolling_hours=(24,)))
    model.fit(frame, fold, [24])
    origins = frame.index[(frame.index >= fold.test_start) & (frame.index < fold.test_start + pd.Timedelta(hours=6))]
    pred = model.predict(frame, origins, [24], "history_proxy")
    assert np.isfinite(pred.yhat).all()
    with pytest.raises(ValueError):
        model.predict(frame, origins, [24], "oracle")

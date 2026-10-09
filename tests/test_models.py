import warnings

import numpy as np
import pandas as pd
import pytest

from gridcast.evaluation import backtest
from gridcast.features.build import FeatureSpec
from gridcast.models.base import fold_origins
from gridcast.models.dshw import DoubleSeasonalHW
from gridcast.models.naive import Persistence, SeasonalNaive, SeasonalProfile
from gridcast.models.sarimax import SarimaxForecaster
from gridcast.models.tabular import LightGBMForecaster, RidgeForecaster

H = [1, 24, 48]


def _mae(pred):
    s = backtest.scoreable(pred)
    return s.assign(ae=(s.y - s.yhat).abs()).groupby("h")["ae"].mean()


def test_seasonal_naive_is_exact_shift(frame, fold):
    m = SeasonalNaive(168)
    p = m.run_fold(frame, fold, [1, 24, 168, 200])
    row = p[(p.h == 200)].iloc[100]
    assert row.yhat == pytest.approx(frame["demand"].loc[row.origin + pd.Timedelta(hours=200 - 336)])
    row = p[(p.h == 24)].iloc[100]
    assert row.yhat == pytest.approx(frame["demand"].loc[row.origin + pd.Timedelta(hours=24 - 168)])


def test_naive_models_ignore_future(frame, fold):
    origins = fold_origins(frame, fold)
    t = origins[300]
    tampered = frame.copy()
    tampered.loc[tampered.index > t, "demand"] = 1.0
    for m in (Persistence(), SeasonalNaive(24), SeasonalNaive(168), SeasonalProfile(4)):
        a = m.predict(frame, origins[:301], H, "none")
        b = m.predict(tampered, origins[:301], H, "none")
        np.testing.assert_allclose(a.yhat, b.yhat)


def test_dshw_beats_weekly_naive_and_is_causal(frame, fold):
    m = DoubleSeasonalHW(fit_years=0.5)
    pred = backtest.run_backtest(lambda: m, frame, [fold], H, verbose=False)[0]
    naive = backtest.run_backtest(lambda: SeasonalNaive(168), frame, [fold], H, verbose=False)[0]
    assert _mae(pred)[1] < _mae(naive)[1]
    origins = fold_origins(frame, fold)[:200]
    tampered = frame.copy()
    tampered.loc[tampered.index > origins[-1], "demand"] *= 2
    np.testing.assert_allclose(m.predict(frame, origins, H, "none").yhat, m.predict(tampered, origins, H, "none").yhat)


def test_sarimax_vectorised_forecast_matches_statsmodels(frame, fold):
    """Z T^h a_{t|t} + x beta must equal refiltering to t and calling statsmodels' own forecast."""
    m = SarimaxForecaster(weather=True, fit_days=40, warmup_days=30, maxiter=15)
    m.fit(frame, fold, [1, 24])
    origins = fold_origins(frame, fold)[:60]
    ours = m.predict(frame, origins, [1, 24], "oracle")
    start = origins[0] - pd.Timedelta(days=m.warmup_days)
    for t in (origins[10], origins[59]):
        idx = frame.index[(frame.index >= start) & (frame.index <= t)]
        mod = m._model(frame, idx)
        fut = pd.date_range(t + pd.Timedelta(hours=1), periods=24, freq="h")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = mod.filter(m.params).get_forecast(24, exog=m._exog(frame, fut).to_numpy(float))
        ref = np.exp(np.asarray(fc.predicted_mean))
        for h in (1, 24):
            got = ours[(ours.origin == t) & (ours.h == h)].yhat.iloc[0]
            assert got == pytest.approx(ref[h - 1], rel=1e-6)
        # model-based 95% interval matches statsmodels' forecast variance
        se = np.asarray(fc.se_mean)
        row = ours[(ours.origin == t) & (ours.h == 24)].iloc[0]
        assert np.log(row["q0.975"] / row["q0.5"]) == pytest.approx(1.959964 * se[23], rel=2e-2)


@pytest.mark.parametrize("cls", [RidgeForecaster, LightGBMForecaster])
def test_tabular_models_learn_and_weather_helps(frame, fold, cls):
    kw = dict(n_estimators=80, q_estimators=30) if cls is LightGBMForecaster else {}
    base = backtest.run_backtest(lambda: cls(FeatureSpec("none"), "b", **kw), frame, [fold], [24], verbose=False)[0]
    wx = backtest.run_backtest(lambda: cls(FeatureSpec("oracle"), "w", **kw), frame, [fold], [24], verbose=False)[0]
    naive = backtest.run_backtest(lambda: SeasonalNaive(168), frame, [fold], [24], verbose=False)[0]
    assert _mae(base)[24] < _mae(naive)[24]
    oracle = _mae(wx[wx.regime == "oracle"])[24]
    forecast = _mae(wx[wx.regime == "forecast"])[24]
    assert oracle < _mae(base)[24]
    assert oracle <= forecast  # forecast weather is noisier than the truth it was trained on
    assert set(wx.regime) == {"forecast", "oracle"}


def test_lightgbm_quantiles_are_sorted_and_cover(frame, fold):
    m = LightGBMForecaster(FeatureSpec("none"), "q", n_estimators=60, q_estimators=60)
    p = backtest.scoreable(backtest.run_backtest(lambda: m, frame, [fold], [24], verbose=False)[0])
    q = p[[c for c in p.columns if c.startswith("q0")]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    cov = ((p.y >= p["q0.05"]) & (p.y <= p["q0.95"])).mean()
    assert 0.6 < cov <= 1.0


def test_blocks_do_not_overlap_training(frame, fold):
    p = backtest.run_backtest(lambda: SeasonalNaive(168), frame, [fold], [1, 48], verbose=False)[0]
    assert p.origin.min() >= fold.cal_start
    cal = p[p.block == "cal"]
    assert (cal.target_time < fold.test_start).all()
    assert (p[p.block == "test"].origin >= fold.test_start).all()

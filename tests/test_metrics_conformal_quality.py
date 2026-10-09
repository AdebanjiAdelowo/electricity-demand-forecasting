import numpy as np
import pandas as pd
import pytest

from gridcast import config
from gridcast.data import quality
from gridcast.evaluation import backtest, metrics
from gridcast.models.naive import SeasonalNaive
from gridcast.probabilistic import conformal


def test_point_metrics_known_values():
    y = np.array([100.0, 200.0, 300.0])
    f = np.array([110.0, 190.0, 330.0])
    assert metrics.mae(y, f) == pytest.approx(50 / 3)
    assert metrics.rmse(y, f) == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert metrics.bias(y, f) == pytest.approx(10.0)
    assert metrics.mape(y, f) == pytest.approx(100 * (0.1 + 0.05 + 0.1) / 3)
    assert metrics.naive_scale(np.arange(400.0), period=168) == pytest.approx(168.0)


def test_pinball_wis_and_coverage_on_gaussian():
    rng = np.random.default_rng(0)
    from scipy.stats import norm

    y = rng.normal(0, 1, 40000)
    lv = np.array(config.QUANTILES)
    q = np.tile(norm.ppf(lv), (len(y), 1))
    for a in config.INTERVAL_ALPHAS:
        cov, width = metrics.interval_stats(y, q, lv, a)
        assert cov == pytest.approx(1 - a, abs=0.01)
        assert width == pytest.approx(2 * norm.ppf(1 - a / 2), rel=1e-6)
    # the calibrated forecast beats a too-narrow and a too-wide one on every proper score
    for fn in (metrics.pinball, metrics.wis, metrics.crps_from_quantiles):
        assert fn(y, q, lv) < fn(y, q * 0.4, lv)
        assert fn(y, q, lv) < fn(y, q * 2.5, lv)
    # CRPS of N(0,1) forecasts for N(0,1) data is 1/sqrt(pi) ~ 0.564; the 9-point grid misses the tails
    assert 0.35 < metrics.crps_from_quantiles(y, q, lv) < 0.6


def test_block_bootstrap_contains_mean():
    rng = np.random.default_rng(1)
    v = rng.normal(5, 1, 5000)
    lo, hi = metrics.block_bootstrap_ci(v, block=50)
    assert lo < 5 < hi and hi - lo < 0.3


def _naive_pred(frame, fold):
    return backtest.run_backtest(lambda: SeasonalNaive(168), frame, [fold], [1, 24], verbose=False)[0]


@pytest.mark.parametrize("mode", ["static", "rolling"])
def test_conformal_reaches_nominal_coverage(frame, fold, mode):
    out = conformal.calibrate(_naive_pred(frame, fold), kind="abs", mode=mode, window=24 * 21)
    out = out[out.y.notna()]
    for a in (0.1, 0.5):
        cov = ((out.y >= out[f"q{a / 2:g}"]) & (out.y <= out[f"q{1 - a / 2:g}"])).mean()
        assert cov == pytest.approx(1 - a, abs=0.07)
    assert set(out.block) == {"test"}
    assert out.method.iloc[0] == f"conformal_{mode}"


def test_rolling_conformal_is_causal(frame, fold):
    """Changing actuals observed after an origin must not change that origin's interval."""
    pred = _naive_pred(frame, fold)
    pred = pred[pred.h == 24]
    cut = fold.test_start + pd.Timedelta(days=10)
    tampered = pred.copy()
    tampered.loc[tampered.target_time > cut, "y"] += 1e6
    a = conformal.calibrate(pred, "abs", "rolling", window=24 * 21)
    b = conformal.calibrate(tampered, "abs", "rolling", window=24 * 21)
    keep = a.origin <= cut
    np.testing.assert_allclose(a.loc[keep, "q0.95"], b.loc[keep, "q0.95"])
    assert not np.allclose(a.loc[~keep, "q0.95"].iloc[-200:], b.loc[~keep, "q0.95"].iloc[-200:])


def test_cqr_static_widens_too_narrow_quantiles(frame, fold):
    pred = _naive_pred(frame, fold)
    for lv in config.QUANTILES:
        pred[f"q{lv:g}"] = pred.yhat + (lv - 0.5) * 10.0  # absurdly narrow
    out = conformal.calibrate(pred, kind="cqr", mode="static")
    cov = ((out.y >= out["q0.05"]) & (out.y <= out["q0.95"])).mean()
    assert cov == pytest.approx(0.9, abs=0.07)


def test_regularise_handles_duplicates_gaps_and_off_grid():
    ts = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 01:30",
                         "2024-01-01 04:00"], utc=True)
    raw = pd.DataFrame({"utc_end": ts, "demand_raw": [1.0, 2.0, 2.5, 9.0, 5.0]})
    df, info = quality.regularise(raw)
    assert info == {"dup": 1, "missing_ts": 2, "off_grid": 1}
    assert len(df) == 5 and df.index.is_monotonic_increasing and df.index.is_unique
    assert df["demand_raw"].loc["2024-01-01 01:00"].item() == 2.5  # the later report wins
    assert df["demand_raw"].isna().sum() == 2


def test_cleaning_is_causal_and_keeps_real_extremes(frame):
    raw = frame[["demand_raw"]].iloc[:24 * 120].reset_index()
    raw.loc[2000:2010, "demand_raw"] = np.nan  # outage
    raw.loc[2500, "demand_raw"] = 0.0  # impossible value
    raw.loc[2600:2620, "demand_raw"] *= 0.7  # a real, large but plausible drop must survive
    clean, rep = quality.clean_demand(raw)
    assert rep.n_missing_values == 11 and rep.n_nonpositive == 1
    assert clean["is_imputed"].sum() == 12
    assert not clean["is_imputed"].iloc[2600:2621].any()
    assert clean["demand"].notna().all()
    # causal: changing the future leaves earlier repaired values unchanged
    raw2 = raw.copy()
    raw2.loc[2100:, "demand_raw"] *= 5
    clean2, _ = quality.clean_demand(raw2)
    np.testing.assert_allclose(clean["demand"].iloc[:2100], clean2["demand"].iloc[:2100])
    # the repair is close to the truth for a short gap
    truth = frame["demand_raw"].iloc[2000:2011].to_numpy()
    assert np.mean(np.abs(clean["demand"].iloc[2000:2011].to_numpy() - truth) / truth) < 0.08

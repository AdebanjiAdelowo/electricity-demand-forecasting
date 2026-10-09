import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from gridcast.evaluation import backtest  # noqa: E402
from gridcast.models.base import fold_origins  # noqa: E402
from gridcast.models.deep import Arrays, DeepForecaster, TrainConfig  # noqa: E402
from gridcast.models.naive import Persistence  # noqa: E402

H = [1, 24, 48]


@pytest.mark.parametrize("arch", ["gru", "tcn", "nbeats"])
def test_deep_models_train_and_beat_persistence(frame, fold, arch):
    cfg = TrainConfig(arch=arch, lookback=48, hidden=32, layers=2, epochs=4, patience=4, device="cpu", weather=True)
    model = DeepForecaster(cfg)
    pred = backtest.run_backtest(lambda: model, frame, [fold], H, verbose=False)[0]
    naive = backtest.run_backtest(Persistence, frame, [fold], H, verbose=False)[0]
    s = backtest.scoreable(pred[pred.regime == "oracle"])
    n = backtest.scoreable(naive)
    assert (s.y - s.yhat).abs().mean() < (n.y - n.yhat).abs().mean()
    assert model.history[-1]["train"] < model.history[0]["train"]
    q = s[[c for c in s.columns if c.startswith("q0")]].to_numpy()
    assert (np.diff(q, axis=1) >= 0).all()
    assert set(pred.regime) == {"forecast", "oracle"}


def test_batch_window_ends_at_origin_and_targets_are_aligned(frame):
    a = Arrays(frame, H, lookback=24, weather=True, device=torch.device("cpu"))
    pos = torch.tensor([5000, 6000])
    past, fut, target, m = a.batch(pos, "oracle")
    y = frame["demand"].to_numpy()
    np.testing.assert_allclose(past[0, :, 0].numpy() * m[0].item(), y[5000 - 23:5001], rtol=1e-5)
    np.testing.assert_allclose(target[0].numpy() * m[0].item(), y[[5001, 5024, 5048]], rtol=1e-5)
    assert past.shape == (2, 24, a.past_dim) and fut.shape == (2, 3, a.fut_dim)


def test_deep_forecast_regime_ignores_future_observed_weather_and_demand(frame, fold):
    cfg = TrainConfig(arch="gru", lookback=48, hidden=16, layers=1, epochs=1, device="cpu", weather=True)
    m = DeepForecaster(cfg)
    m.fit(frame, fold, H)
    origins = fold_origins(frame, fold)[:50]
    a = m.predict(frame, origins, H, "forecast")
    tampered = frame.copy()
    after = tampered.index > origins[-1]
    tampered.loc[after, "demand"] *= 3
    tampered.loc[after, ["temp", "rh", "wind"]] += 20
    m.arr = Arrays(tampered, H, 48, True, torch.device("cpu"))
    b = m.predict(tampered, origins, H, "forecast")
    last = a.origin == origins[-1]
    np.testing.assert_allclose(a.yhat[last], b.yhat[last], rtol=1e-5)
    # while the oracle regime is, by construction, sensitive to it
    c = m.predict(tampered, origins, H, "oracle")
    m.arr = Arrays(frame, H, 48, True, torch.device("cpu"))
    d = m.predict(frame, origins, H, "oracle")
    assert not np.allclose(c.yhat[last], d.yhat[last])
    assert isinstance(a.origin.iloc[0], pd.Timestamp)

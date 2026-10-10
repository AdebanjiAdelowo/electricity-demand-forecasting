"""Double seasonal Holt-Winters exponential smoothing with AR(1) error correction.

Taylor (2003), "Short-term electricity demand forecasting using double seasonal
exponential smoothing", J. Oper. Res. Soc. 54. Applied to log demand, which
makes the daily (24 h) and weekly (168 h) seasonal effects multiplicative.

    e_t = y_t - (l_{t-1} + d_{t-24} + w_{t-168})
    l_t = l_{t-1} + alpha * e_t
    d_t = d_{t-24} + delta * (1 - alpha) * e_t
    w_t = w_{t-168} + omega * (1 - alpha) * (1 - delta) * e_t
    yhat_t(h) = l_t + d_{t+h-24*ceil(h/24)} + w_{t+h-168*ceil(h/168)} + phi^h * e_t

One recursive pass yields the state at every hour, so forecasts for every
origin and horizon are obtained by indexing, without refitting per origin.

The four parameters are estimated by minimising the mean absolute log error
averaged over several horizons (1, 24, 48, 168 h by default). Fitting on
one-step errors alone drives alpha towards 1, which is excellent one hour
ahead and worse than a daily naive forecast one day ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gridcast.models.base import Forecaster, fit_slice, long_frame

S1, S2 = 24, 168


def dshw_filter(z: np.ndarray, alpha: float, delta: float, omega: float):
    """Run the recursion over log-demand z. Returns level, daily, weekly states and 1-step errors."""
    n = len(z)
    zl = z.tolist()
    init = z[: 2 * S2]
    lvl = float(init.mean())
    week = (init[:S2] + init[S2:]) / 2 - lvl
    day0 = week.reshape(7, 24).mean(axis=0)
    d = [0.0] * n
    w = [0.0] * n
    l_ = [0.0] * n
    e = [0.0] * n
    dbuf = day0.tolist()  # d_{t-24}, indexed by t % 24
    wbuf = (week - np.tile(day0, 7)).tolist()  # w_{t-168}, indexed by t % 168
    ca = 1 - alpha
    for t in range(n):
        i1 = t % S1
        i2 = t % S2
        err = zl[t] - (lvl + dbuf[i1] + wbuf[i2])
        lvl = lvl + alpha * err
        dbuf[i1] = dbuf[i1] + delta * ca * err
        wbuf[i2] = wbuf[i2] + omega * ca * (1 - delta) * err
        l_[t] = lvl
        d[t] = dbuf[i1]
        w[t] = wbuf[i2]
        e[t] = err
    return np.array(l_), np.array(d), np.array(w), np.array(e)


def _unpack(theta):
    s = 1 / (1 + np.exp(-np.asarray(theta)))
    return s[0], s[1], s[2], s[3]


class DoubleSeasonalHW(Forecaster):
    name = "dshw"
    family = "statistical"
    strategy = "recursive"

    def __init__(self, fit_years: float = 2.0, loss_horizons=(1, 24, 48, 168)):
        super().__init__()
        self.fit_years = fit_years
        self.loss_horizons = tuple(loss_horizons)
        self.params: tuple[float, float, float, float] | None = None

    def fit(self, df, fold, horizons):
        z = np.log(fit_slice(df, fold)["demand"].to_numpy(float))
        z = z[-int(self.fit_years * 8766) - 2 * S2:]

        hs = np.array(self.loss_horizons)
        t0 = 2 * S2
        pos = np.arange(t0, len(z) - hs.max())

        def loss(theta):
            a, dl, om, phi = _unpack(theta)
            lvl, d, w, e = dshw_filter(z, a, dl, om)
            tot = 0.0
            for h in hs:
                zhat = lvl[pos] + d[pos + h - S1 * (-(-h // S1))] + w[pos + h - S2 * (-(-h // S2))] + (phi**h) * e[pos]
                tot += float(np.mean(np.abs(z[pos + h] - zhat)))
            return tot / len(hs)

        res = minimize(loss, x0=[-2.0, -1.0, -1.0, 2.0], method="Nelder-Mead",
                       options={"maxiter": 250, "xatol": 1e-3, "fatol": 1e-10})
        self.params = _unpack(res.x)
        self.info.n_params = 4
        self.info.extra = dict(zip(["alpha", "delta", "omega", "phi"], map(float, self.params), strict=True))

    def predict(self, df, origins, horizons, regime):
        a, dl, om, phi = self.params
        # warm the recursion up on the two years before the first origin
        start = origins[0] - pd.Timedelta(days=730)
        sub = df.loc[(df.index >= start) & (df.index <= origins[-1]), "demand"]
        z = np.log(sub.to_numpy(float))
        lvl, d, w, e = dshw_filter(z, a, dl, om)
        pos = sub.index.get_indexer(origins)
        frames = []
        for h in horizons:
            dj = pos + h - S1 * (-(-h // S1))
            wk = pos + h - S2 * (-(-h // S2))
            zhat = lvl[pos] + d[dj] + w[wk] + (phi**h) * e[pos]
            frames.append(long_frame(origins, h, regime, np.exp(zhat)))
        return pd.concat(frames, ignore_index=True)

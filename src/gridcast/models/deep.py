"""Multi-output neural forecasters (PyTorch): GRU, TCN and an N-BEATS-style MLP.

Each network maps a lookback window ending at the forecast origin, plus
covariates that are known for the target hours, to all horizons and all
quantile levels at once (multi-output, trained with the pinball loss). The
0.5 quantile is used as the point forecast.

Inputs
    past  (L, C): demand / level, [observed temperature], calendar encodings
    future (H, F): calendar at each target hour, [weather for the target hour]
The level is the mean demand of the 168 hours ending at the origin; targets
are divided by it. Weather for the target hour is reanalysis during training
and either reanalysis ("oracle") or archived NWP ("forecast") at test time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from gridcast import config
from gridcast.features.build import BASE_TEMP_C
from gridcast.features.calendar import calendar_frame
from gridcast.models.base import Forecaster, long_frame

CAL_PAST = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_offday"]
CAL_FUT = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_offday", "is_holiday", "doy_sin", "doy_cos"]


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _wx_block(temp, rh, wind) -> np.ndarray:
    t = np.asarray(temp, dtype=np.float32)
    return np.stack(
        [(t - 20.0) / 10.0, np.clip(t - BASE_TEMP_C, 0, None) / 10.0, np.clip(BASE_TEMP_C - t, 0, None) / 10.0,
         np.asarray(rh, dtype=np.float32) / 100.0, np.asarray(wind, dtype=np.float32) / 5.0], axis=1)


class Arrays:
    """Device tensors for the whole series; windows are gathered by index."""

    def __init__(self, df: pd.DataFrame, horizons: list[int], lookback: int, weather: bool, device: torch.device):
        self.index = df.index
        self.h = torch.tensor(horizons, device=device)
        self.lookback = lookback
        self.weather = weather
        self.device = device
        y = df["demand"].to_numpy(np.float32)
        m = df["demand"].rolling(168, min_periods=168).mean().to_numpy(np.float32)
        cal = calendar_frame(df.index)
        self.y = torch.tensor(y, device=device)
        self.m = torch.tensor(m, device=device)
        past = [cal[CAL_PAST].to_numpy(np.float32)]
        if weather:
            past.append(((df["temp"].ffill().to_numpy(np.float32) - 20.0) / 10.0)[:, None])
        self.past_cov = torch.tensor(np.concatenate(past, axis=1), device=device)
        self.cal_fut = torch.tensor(cal[CAL_FUT].to_numpy(np.float32), device=device)
        self.n = len(df)
        if weather:
            self.wx = {"oracle": torch.tensor(_wx_block(df["temp"], df["rh"], df["wind"])[None], device=device)}
            if "temp_d1" in df:
                leads = [_wx_block(df[f"temp_d{d}"], df[f"rh_d{d}"], df[f"wind_d{d}"]) for d in config.FORECAST_LEAD_DAYS]
                self.wx["forecast"] = torch.tensor(np.stack(leads), device=device)
            self.lead_idx = torch.tensor([config.forecast_lead_day(h) - 1 for h in horizons], device=device)
        valid = np.isfinite(m)
        valid[: max(lookback, 168)] = False
        self.valid_origin = valid

    @property
    def past_dim(self) -> int:
        return 1 + self.past_cov.shape[1]

    @property
    def fut_dim(self) -> int:
        return self.cal_fut.shape[1] + (5 if self.weather else 0)

    def batch(self, pos: torch.Tensor, regime: str = "oracle", with_target: bool = True):
        L = self.lookback
        win = pos[:, None] + torch.arange(-L + 1, 1, device=self.device)[None, :]
        m = self.m[pos]
        past = torch.cat([(self.y[win] / m[:, None])[..., None], self.past_cov[win]], dim=-1)
        tgt = pos[:, None] + self.h[None, :]
        tgt_c = tgt.clamp(max=self.n - 1)
        fut = self.cal_fut[tgt_c]
        if self.weather:
            wx = self.wx[regime]
            lead = torch.zeros_like(self.lead_idx) if wx.shape[0] == 1 else self.lead_idx
            w = wx[lead[None, :].expand_as(tgt_c), tgt_c]
            # a missing forecast value falls back to the last observation at the origin
            obs = self.wx["oracle"][0][pos][:, None, :].expand_as(w)
            w = torch.where(torch.isnan(w), obs, w)
            fut = torch.cat([fut, torch.nan_to_num(w)], dim=-1)
        target = (self.y[tgt_c] / m[:, None]) if with_target else None
        return past, fut, target, m


class Decoder(nn.Module):
    """Shared head: (context, future covariates of one horizon, horizon embedding) -> quantiles."""

    def __init__(self, ctx: int, fut_dim: int, n_h: int, n_q: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(n_h, 8)
        self.net = nn.Sequential(nn.Linear(ctx + fut_dim + 8, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_q))

    def forward(self, ctx, fut):
        B, H, _ = fut.shape
        e = self.emb.weight[None].expand(B, H, -1)
        x = torch.cat([ctx[:, None, :].expand(B, H, -1), fut, e], dim=-1)
        return self.net(x)


class GRUNet(nn.Module):
    def __init__(self, past_dim, fut_dim, n_h, n_q, hidden=96, layers=2, dropout=0.1, **_):
        super().__init__()
        self.enc = nn.GRU(past_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.dec = Decoder(hidden, fut_dim, n_h, n_q, hidden)

    def forward(self, past, fut):
        _, hN = self.enc(past)
        return 1.0 + self.dec(hN[-1], fut)


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.c2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.pad = pad
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        o = self.drop(torch.relu(self.c1(x)[..., : -self.pad]))  # chomp keeps the convolution causal
        o = self.drop(torch.relu(self.c2(o)[..., : -self.pad]))
        return torch.relu(o + self.skip(x))


class TCNNet(nn.Module):
    def __init__(self, past_dim, fut_dim, n_h, n_q, hidden=64, layers=6, kernel=3, dropout=0.1, **_):
        super().__init__()
        blocks = [_TCNBlock(past_dim if i == 0 else hidden, hidden, kernel, 2**i, dropout) for i in range(layers)]
        self.tcn = nn.Sequential(*blocks)
        self.dec = Decoder(hidden, fut_dim, n_h, n_q, hidden)

    def forward(self, past, fut):
        z = self.tcn(past.transpose(1, 2))[..., -1]
        return 1.0 + self.dec(z, fut)


class _NBeatsBlock(nn.Module):
    def __init__(self, in_dim, back_dim, out_dim, width, depth):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.fc = nn.Sequential(*layers)
        self.back = nn.Linear(width, back_dim)
        self.fore = nn.Linear(width, out_dim)

    def forward(self, x):
        z = self.fc(x)
        return self.back(z), self.fore(z)


class NBeatsNet(nn.Module):
    """Generic-basis N-BEATS (Oreshkin et al. 2020) with exogenous inputs appended to every block
    (in the spirit of NBEATSx, Olivares et al. 2023). Backcast residuals are taken on the demand window."""

    def __init__(self, past_dim, fut_dim, n_h, n_q, lookback, hidden=256, layers=4, depth=3, **_):
        super().__init__()
        self.lookback, self.n_h, self.n_q = lookback, n_h, n_q
        exo = lookback * (past_dim - 1) + n_h * fut_dim
        self.blocks = nn.ModuleList(
            [_NBeatsBlock(lookback + exo, lookback, n_h * n_q, hidden, depth) for _ in range(layers)])

    def forward(self, past, fut):
        B = past.shape[0]
        resid = past[..., 0] - 1.0
        exo = torch.cat([past[..., 1:].reshape(B, -1), fut.reshape(B, -1)], dim=1)
        out = 0.0
        for blk in self.blocks:
            back, fore = blk(torch.cat([resid, exo], dim=1))
            resid = resid - back
            out = out + fore
        return 1.0 + out.view(B, self.n_h, self.n_q)


ARCH = {"gru": GRUNet, "tcn": TCNNet, "nbeats": NBeatsNet}


def pinball_loss(pred, target, levels):
    d = target[..., None] - pred
    return torch.maximum(levels * d, (levels - 1) * d).mean()


@dataclass
class TrainConfig:
    arch: str = "gru"
    lookback: int = 168
    hidden: int = 96
    layers: int = 2
    weather: bool = True
    epochs: int = 30
    patience: int = 4
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 0
    device: str | None = None


class DeepForecaster(Forecaster):
    family = "deep"
    strategy = "multi-output"
    probabilistic = True

    def __init__(self, cfg: TrainConfig | None = None, name: str | None = None, **kw):
        super().__init__()
        self.cfg = cfg or TrainConfig(**kw)
        self.name = name or (self.cfg.arch + ("_weather" if self.cfg.weather else ""))
        self.regimes = ("forecast", "oracle") if self.cfg.weather else ("none",)
        self.net: nn.Module | None = None
        self.arr: Arrays | None = None
        self.history: list[dict] = []

    def _positions(self, start, end, hmax: int, need_target: bool = True) -> np.ndarray:
        a = self.arr
        idx = a.index
        ok = a.valid_origin & np.asarray(idx >= start)
        ok &= np.asarray(idx + pd.Timedelta(hours=hmax) < end) if need_target else np.asarray(idx < end)
        return np.flatnonzero(ok)

    def fit(self, df, fold, horizons):
        c = self.cfg
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        dev = pick_device(c.device)
        self.horizons = list(horizons)
        self.arr = a = Arrays(df, self.horizons, c.lookback, c.weather, dev)
        levels = torch.tensor(config.QUANTILES, device=dev, dtype=torch.float32)
        net_cls = ARCH[c.arch]
        self.net = net_cls(a.past_dim, a.fut_dim, len(horizons), len(config.QUANTILES), lookback=c.lookback,
                           hidden=c.hidden, layers=c.layers).to(dev)
        self.info.n_params = int(sum(p.numel() for p in self.net.parameters()))
        hmax = max(horizons)
        tr = torch.tensor(self._positions(fold.train_start, fold.val_start, hmax), device=dev)
        va = torch.tensor(self._positions(fold.val_start, fold.cal_start, hmax), device=dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
        best, best_state, bad = float("inf"), None, 0
        g = torch.Generator(device="cpu").manual_seed(c.seed)
        for epoch in range(c.epochs):
            self.net.train()
            perm = tr[torch.randperm(len(tr), generator=g).to(dev)]
            tot, nb = 0.0, 0
            for i in range(0, len(perm), c.batch_size):
                past, fut, target, _ = a.batch(perm[i:i + c.batch_size], "oracle")
                loss = pinball_loss(self.net(past, fut), target, levels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach())
                nb += 1
            vloss = self._eval_loss(va, levels)
            sched.step(vloss)
            self.history.append({"epoch": epoch, "train": tot / max(nb, 1), "val": vloss})
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= c.patience:
                    break
        self.net.load_state_dict(best_state)
        self.info.extra = {"epochs_run": len(self.history), "best_val_pinball": best, "device": str(dev)}

    @torch.no_grad()
    def _eval_loss(self, pos, levels) -> float:
        self.net.eval()
        tot, n = 0.0, 0
        for i in range(0, len(pos), 2048):
            past, fut, target, _ = self.arr.batch(pos[i:i + 2048], "oracle")
            tot += float(pinball_loss(self.net(past, fut), target, levels)) * len(past)
            n += len(past)
        return tot / max(n, 1)

    @torch.no_grad()
    def predict(self, df, origins, horizons, regime):
        a = self.arr
        self.net.eval()
        pos_np = a.index.get_indexer(origins)
        pos = torch.tensor(pos_np, device=a.device)
        wx_regime = regime if regime in ("oracle", "forecast") else "oracle"
        outs = []
        for i in range(0, len(pos), 2048):
            past, fut, _, m = a.batch(pos[i:i + 2048], wx_regime, with_target=False)
            outs.append((self.net(past, fut) * m[:, None, None]).float().cpu().numpy())
        q = np.concatenate(outs)  # (n, H, Q)
        med = config.QUANTILES.index(0.5)
        frames = []
        for h in horizons:
            j = self.horizons.index(h)
            qq = np.sort(q[:, j, :], axis=1)
            frames.append(long_frame(origins, h, regime, qq[:, med], qq))
        return pd.concat(frames, ignore_index=True)

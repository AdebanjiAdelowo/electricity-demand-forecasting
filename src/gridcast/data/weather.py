"""Weather loaders (Open-Meteo, CC BY 4.0).

Two distinct products are downloaded and never mixed silently:

* reanalysis  : ERA5-based historical archive. This is *observed-like* weather.
                Using it at the target time is an oracle experiment.
* nwp forecast: archived GFS forecasts from the Previous Runs API, indexed by
                valid time and lead bucket (1..7 days before valid time). This
                is what a forecaster could actually have had.

All series are aggregated over fixed load-centre points with fixed weights and
converted from instantaneous values at hour marks to hour-ending interval
means (average of the two bounding instants) so they align with EIA-930.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3.util.connection as _conn

from gridcast import config

# Open-Meteo resolves to IPv6 first on some networks and then stalls for the
# full connect timeout; IPv4 answers immediately.
_conn.allowed_gai_family = lambda: __import__("socket").AF_INET

SHORT = {"temperature_2m": "temp", "relative_humidity_2m": "rh", "wind_speed_10m": "wind"}


def _get(url: str, params: dict, retries: int = 6) -> dict:
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(15 * (attempt + 1))
            continue
        raise RuntimeError(f"Open-Meteo {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"Open-Meteo failed after {retries} retries: {url}")


def _frame(payload: dict) -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")


def download_reanalysis(start: str, end: str, dest: Path | None = None, pause: float = 1.0) -> Path:
    dest = dest or config.DATA_RAW / "weather_reanalysis_points.parquet"
    if dest.exists():
        return dest
    years = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
    frames = []
    for name, (lat, lon, _) in config.WEATHER_POINTS.items():
        parts = []
        for y in years:
            s = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")).date().isoformat()
            e = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31")).date().isoformat()
            p = _get(
                config.OPEN_METEO_ARCHIVE,
                dict(latitude=lat, longitude=lon, start_date=s, end_date=e,
                     hourly=",".join(config.WEATHER_VARS), timezone="UTC", wind_speed_unit="ms"),
            )
            parts.append(_frame(p))
            time.sleep(pause)
        f = pd.concat(parts).rename(columns=SHORT)
        f.columns = [f"{c}__{name}" for c in f.columns]
        frames.append(f)
    out = pd.concat(frames, axis=1).sort_index()
    out.to_parquet(dest)
    _write_meta(dest, "ERA5-based Open-Meteo Historical Weather API", start, end)
    return dest


def download_nwp_forecasts(start: str, end: str, dest: Path | None = None, pause: float = 1.0) -> Path:
    dest = dest or config.DATA_RAW / "weather_nwp_points.parquet"
    if dest.exists():
        return dest
    hourly = [f"{v}_previous_day{d}" for v in config.WEATHER_VARS for d in config.FORECAST_LEAD_DAYS]
    edges = list(pd.date_range(start, end, freq="QS")) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    if edges[0] > pd.Timestamp(start):
        edges = [pd.Timestamp(start)] + edges
    frames = []
    for name, (lat, lon, _) in config.WEATHER_POINTS.items():
        parts = []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            p = _get(
                config.OPEN_METEO_PREVIOUS_RUNS,
                dict(latitude=lat, longitude=lon, start_date=a.date().isoformat(),
                     end_date=(b - pd.Timedelta(days=1)).date().isoformat(), hourly=",".join(hourly),
                     models=config.NWP_MODEL, timezone="UTC", wind_speed_unit="ms"),
            )
            parts.append(_frame(p))
            time.sleep(pause)
        f = pd.concat(parts)
        f.columns = [
            f"{SHORT[c.split('_previous_day')[0]]}_d{c.split('_previous_day')[1]}__{name}" for c in f.columns
        ]
        frames.append(f)
    out = pd.concat(frames, axis=1).sort_index()
    out.to_parquet(dest)
    _write_meta(dest, f"Open-Meteo Previous Runs API ({config.NWP_MODEL})", start, end)
    return dest


def _write_meta(dest: Path, product: str, start: str, end: str) -> None:
    meta = {
        "product": product,
        "license": "CC BY 4.0 (Open-Meteo.com)",
        "start": start,
        "end": end,
        "points": {k: {"lat": v[0], "lon": v[1], "weight": v[2]} for k, v in config.WEATHER_POINTS.items()},
        "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def _to_hour_ending_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Instantaneous values at t-1h and t -> mean for the hour ending at t."""
    return (df + df.shift(1)) / 2.0


def aggregate_points(points: pd.DataFrame) -> pd.DataFrame:
    """Weighted aggregate across load centres plus cross-point temperature extremes.

    Input columns are `<var>__<point>` (reanalysis) or `<var>_d<k>__<point>` (NWP).
    Weights are renormalised over the points that are present at each hour.
    """
    names = list(config.WEATHER_POINTS)
    w = np.array([config.WEATHER_POINTS[n][2] for n in names])
    out = {}
    for var in sorted({c.split("__")[0] for c in points.columns}):
        block = points[[f"{var}__{n}" for n in names]].to_numpy(dtype=float)
        ok = ~np.isnan(block)
        wsum = (ok * w).sum(axis=1)
        agg = np.where(wsum > 0.5, np.nansum(block * w, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
        out[var] = agg
        if var.startswith("temp"):
            with np.errstate(all="ignore"):
                all_nan = ~ok.any(axis=1)
                out[var.replace("temp", "tmin", 1)] = np.where(all_nan, np.nan, np.nanmin(np.where(ok, block, np.inf), axis=1))
                out[var.replace("temp", "tmax", 1)] = np.where(all_nan, np.nan, np.nanmax(np.where(ok, block, -np.inf), axis=1))
    agg = pd.DataFrame(out, index=points.index)
    return _to_hour_ending_mean(agg)


NWP_VARS = ("temp", "rh", "wind", "tmin", "tmax")


def debias_nwp(df: pd.DataFrame, days: int = 28, min_days: int = 7) -> pd.DataFrame:
    """Causal bias correction of archived NWP forecasts against the observation proxy.

    The archived GFS product is systematically offset from the ERA5-based
    series the models are trained on (about +1.4 C and -10 points of relative
    humidity from lead day 2 onward). For each variable, lead day d and UTC hour
    of day, the correction at valid time t is the mean forecast-minus-observed
    difference over the `days` most recent days ending d days before t. Those
    observations precede every origin that is allowed to use lead d, so the
    corrected value `<var>_c<d>` is available in real time. Until `min_days`
    of pairs exist the raw forecast is passed through.
    """
    out = {}
    hour = df.index.hour
    for v in NWP_VARS:
        for d in config.FORECAST_LEAD_DAYS:
            raw = df[f"{v}_d{d}"]
            err = raw - df[v]
            bias = err.groupby(hour).transform(lambda s, d=d: s.rolling(days, min_periods=min_days).mean().shift(d))
            c = raw - bias.fillna(0.0)
            if v == "rh":
                c = c.clip(0, 100)
            if v == "wind":
                c = c.clip(lower=0)
            out[f"{v}_c{d}"] = c
    return pd.DataFrame(out, index=df.index)

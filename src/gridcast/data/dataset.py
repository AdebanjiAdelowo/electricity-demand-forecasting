"""Assemble the processed hourly study dataset and its manifest."""

from __future__ import annotations

import json

import pandas as pd

from gridcast import config
from gridcast.data import eia, quality, weather

PROCESSED = config.DATA_PROCESSED / "ercot_hourly.parquet"
MANIFEST = config.DATA_PROCESSED / "manifest.json"


def build(download: bool = True) -> pd.DataFrame:
    config.ensure_dirs()
    if download:
        eia.download()
        weather.download_reanalysis("2015-06-25", "2026-09-05")
        weather.download_nwp_forecasts("2024-01-01", "2026-09-05")
    raw = eia.read_raw()
    raw = raw[(raw.utc_end >= config.DATA_START) & (raw.utc_end < config.DATA_END)]
    df, report = quality.clean_demand(raw)
    df.index.name = "utc_end"

    rea = weather.aggregate_points(pd.read_parquet(config.DATA_RAW / "weather_reanalysis_points.parquet"))
    nwp = weather.aggregate_points(pd.read_parquet(config.DATA_RAW / "weather_nwp_points.parquet"))
    out = df.join(rea, how="left").join(nwp, how="left")
    out = out.drop(columns=["net_generation", "total_interchange"])
    out.to_parquet(PROCESSED)

    eia_meta_path = (config.DATA_RAW / eia.RAW_NAME).with_suffix(".meta.json")
    manifest = {
        "dataset_version": f"eia930-TEX+openmeteo@{pd.Timestamp.now(tz='UTC').date().isoformat()}",
        "region": config.REGION_NAME,
        "time_index": "UTC, hour ending",
        "local_time_zone": config.LOCAL_TZ,
        "rows": len(out),
        "start": str(out.index.min()),
        "end": str(out.index.max()),
        "eia_raw_sha256": eia.sha256(config.DATA_RAW / eia.RAW_NAME),
        "eia_source": json.loads(eia_meta_path.read_text()) if eia_meta_path.exists() else {"url": config.EIA_URL},
        "weather_points": {k: {"lat": v[0], "lon": v[1], "weight": v[2]} for k, v in config.WEATHER_POINTS.items()},
        "nwp_model": config.NWP_MODEL,
        "nwp_first_valid": {c: str(out[c].first_valid_index()) for c in ("temp_d1", "temp_d7")},
        "weather_missing_fraction": {c: float(out[c].isna().mean()) for c in ("temp", "rh", "wind")},
        "quality": report.to_dict(),
        "licenses": {
            "demand": "U.S. EIA, Form EIA-930, public domain (U.S. government work)",
            "weather": "Open-Meteo.com, CC BY 4.0; reanalysis derived from Copernicus ERA5, forecasts from NOAA GFS",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return out


def load() -> pd.DataFrame:
    if not PROCESSED.exists():
        raise FileNotFoundError(f"{PROCESSED} is missing. Run `gridcast data` first.")
    df = pd.read_parquet(PROCESSED)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df

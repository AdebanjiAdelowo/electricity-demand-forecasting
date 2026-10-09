"""EIA-930 hourly demand loader for the Texas region (ERCOT).

Source: U.S. Energy Information Administration, Hourly Electric Grid Monitor
(Form EIA-930). U.S. government data, public domain.

Timestamps in the source are *hour ending*. We keep that convention and index
everything by the UTC hour-ending timestamp, which has no daylight-saving gaps
or duplicates. Local time is only ever derived from UTC.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from gridcast import config

RAW_NAME = "Region_TEX.xlsx"
SHEET = "Published Hourly Data"
COLUMNS = {
    "UTC time": "utc_end",
    "Demand": "demand_raw",
    "Demand forecast": "eia_day_ahead_forecast",
    "Net generation": "net_generation",
    "Total interchange": "total_interchange",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path | None = None, force: bool = False) -> Path:
    dest = dest or config.DATA_RAW / RAW_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    r = requests.get(config.EIA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=600)
    r.raise_for_status()
    dest.write_bytes(r.content)
    meta = {
        "url": config.EIA_URL,
        "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "last_modified": r.headers.get("Last-Modified"),
        "sha256": sha256(dest),
        "bytes": dest.stat().st_size,
    }
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return dest


def read_raw(path: Path | None = None) -> pd.DataFrame:
    """Parse the workbook into a frame with one row per reported hour (no fixes applied)."""
    path = path or config.DATA_RAW / RAW_NAME
    df = pd.read_excel(path, sheet_name=SHEET, usecols=list(COLUMNS), engine="openpyxl")
    df = df.rename(columns=COLUMNS)
    df["utc_end"] = pd.to_datetime(df["utc_end"], utc=True)
    for c in df.columns.drop("utc_end"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

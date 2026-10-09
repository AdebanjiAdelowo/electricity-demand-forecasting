"""Timestamp regularisation and causal data cleaning for the demand series.

Rules are deliberately conservative: real extreme events (for example the
February 2021 load shed) must survive cleaning, so only physically implausible
values are flagged. Every repair is *causal* (uses the past only), which lets
the cleaned series be used as a model input at any forecast origin without
leaking future information. Repaired hours are flagged and are excluded from
scoring when they are forecast targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    n_rows_in: int
    n_duplicate_timestamps: int
    n_missing_timestamps: int
    n_off_grid_timestamps: int
    n_missing_values: int
    n_nonpositive: int
    n_out_of_range: int
    n_stuck: int
    n_imputed: int
    longest_gap_hours: int
    first: str
    last: str

    def to_dict(self) -> dict:
        return asdict(self)


def regularise(df: pd.DataFrame, time_col: str = "utc_end") -> tuple[pd.DataFrame, dict]:
    """Return a frame on a complete hourly UTC grid.

    Handles off-grid stamps (dropped), duplicate hours (last report wins, as in
    a revision) and missing hours (inserted as NaN rows).
    """
    df = df.copy()
    ts = pd.to_datetime(df[time_col], utc=True)
    on_grid = (ts.dt.minute == 0) & (ts.dt.second == 0)
    n_off = int((~on_grid).sum())
    df = df.loc[on_grid].assign(**{time_col: ts[on_grid]}).sort_values(time_col, kind="stable")
    n_dup = int(df[time_col].duplicated(keep="last").sum())
    df = df.drop_duplicates(time_col, keep="last").set_index(time_col)
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC", name=time_col)
    n_missing = len(grid) - len(df)
    return df.reindex(grid), {"dup": n_dup, "missing_ts": n_missing, "off_grid": n_off}


def flag_implausible(y: pd.Series, window: int = 24 * 365, lo: float = 0.5, hi: float = 1.5, stuck: int = 6):
    """Causal plausibility flags. Returns dict of boolean Series."""
    nonpos = y <= 0
    ok = y.where(~nonpos)
    trail_max = ok.shift(1).rolling(window, min_periods=24 * 28).max()
    trail_min = ok.shift(1).rolling(window, min_periods=24 * 28).min()
    out_of_range = (ok > hi * trail_max) | (ok < lo * trail_min)
    same = ok.diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    stuck_flag = run >= (stuck - 1)
    return {"nonpositive": nonpos.fillna(False), "out_of_range": out_of_range.fillna(False), "stuck": stuck_flag}


def causal_seasonal_fill(y: pd.Series, bad: pd.Series, period: int = 168, level_window: int = 24) -> pd.Series:
    """Replace bad/missing values using only earlier data.

    fill(t) = clean(t - period) * r, where r is the ratio between the mean of
    the last `level_window` clean hours before the current gap and the mean of
    the same hours one period earlier (a causal level adjustment, clipped to
    [0.7, 1.4]). Falls back to earlier periods, then to the last value.
    """
    v = y.to_numpy(dtype=float).copy()
    b = bad.to_numpy() | np.isnan(v)
    v[b] = np.nan
    idx = np.flatnonzero(b)
    ratio = 1.0
    prev = -10
    for i in idx:
        if i != prev + 1:  # new gap: recompute level ratio from the hours just before it
            a = v[max(i - level_window, 0):i]
            c = v[max(i - level_window - period, 0):max(i - period, 0)]
            if len(a) == len(c) and len(a) > 0 and np.isfinite(a).all() and np.isfinite(c).all():
                ratio = float(np.clip(a.mean() / c.mean(), 0.7, 1.4))
            else:
                ratio = 1.0
        prev = i
        fill = np.nan
        for k in (1, 2, 3, 4):
            j = i - k * period
            if j >= 0 and np.isfinite(v[j]):
                fill = v[j] * ratio
                break
        if not np.isfinite(fill):
            fill = v[i - 1] if i > 0 and np.isfinite(v[i - 1]) else np.nan
        v[i] = fill
    return pd.Series(v, index=y.index, name=y.name)


def clean_demand(raw: pd.DataFrame, value_col: str = "demand_raw") -> tuple[pd.DataFrame, QualityReport]:
    n_in = len(raw)
    df, info = regularise(raw)
    y = df[value_col]
    flags = flag_implausible(y)
    missing = y.isna()
    bad = missing | flags["nonpositive"] | flags["out_of_range"] | flags["stuck"]
    df["demand"] = causal_seasonal_fill(y, bad)
    df["is_imputed"] = bad
    gaps = bad.astype(int).groupby((~bad).cumsum()).sum()
    report = QualityReport(
        n_rows_in=n_in,
        n_duplicate_timestamps=info["dup"],
        n_missing_timestamps=info["missing_ts"],
        n_off_grid_timestamps=info["off_grid"],
        n_missing_values=int(missing.sum()),
        n_nonpositive=int(flags["nonpositive"].sum()),
        n_out_of_range=int(flags["out_of_range"].sum()),
        n_stuck=int(flags["stuck"].sum()),
        n_imputed=int(bad.sum()),
        longest_gap_hours=int(gaps.max()) if len(gaps) else 0,
        first=str(df.index.min()),
        last=str(df.index.max()),
    )
    return df, report

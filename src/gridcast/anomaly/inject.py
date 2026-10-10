"""Synthetic anomaly injection.

There are no anomaly labels for ERCOT demand, so detector accuracy can only be
measured against anomalies we plant ourselves. Everything produced here is
SIMULATED; results on it say how sensitive a detector is to a given shape and
size of disturbance, not how often real grid events are caught.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TYPES = ["spike", "dip_block", "level_shift", "flatline"]
DURATION = {"spike": (1, 1), "dip_block": (3, 12), "level_shift": (48, 168), "flatline": (6, 24)}


@dataclass
class Injection:
    df: pd.DataFrame  # contaminated copy
    labels: pd.Series  # bool per hour
    events: pd.DataFrame  # id, type, magnitude, start, end (inclusive positions and timestamps)


def inject(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, kind: str, magnitude: float, n_events: int = 40,
           seed: int = 0, gap: int = 72) -> Injection:
    """Plant `n_events` non-overlapping anomalies of one type and size inside [start, end).

    magnitude is the relative size (0.05 = 5 percent of the true value). Spikes
    and level shifts take a random sign, dips are negative, flatlines hold the
    last good value and ignore magnitude.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    y = out["demand"].to_numpy(float).copy()
    lo, hi = df.index.searchsorted(start), df.index.searchsorted(end)
    dmin, dmax = DURATION[kind]
    labels = np.zeros(len(df), dtype=bool)
    taken = np.zeros(len(df), dtype=bool)
    events = []
    tries = 0
    while len(events) < n_events and tries < 50 * n_events:
        tries += 1
        dur = int(rng.integers(dmin, dmax + 1))
        s = int(rng.integers(lo, hi - dur - gap))
        if taken[max(s - gap, 0): s + dur + gap].any():
            continue
        e = s + dur
        if kind == "spike":
            y[s:e] *= 1 + magnitude * rng.choice([-1, 1])
        elif kind == "dip_block":
            y[s:e] *= 1 - magnitude
        elif kind == "level_shift":
            y[s:e] *= 1 + magnitude * rng.choice([-1, 1])
        elif kind == "flatline":
            y[s:e] = y[s - 1]
        else:
            raise ValueError(kind)
        labels[s:e] = True
        taken[s:e] = True
        events.append({"id": len(events), "type": kind, "magnitude": magnitude, "start_pos": s, "end_pos": e - 1,
                       "start": df.index[s], "end": df.index[e - 1], "hours": dur})
    out["demand"] = y
    return Injection(out, pd.Series(labels, index=df.index, name="is_anomaly"), pd.DataFrame(events))

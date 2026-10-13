"""Timestamp-gated rolling calibration and bounded delayed ACI.

No row-count shift is used. Actual feedback is admitted at target_time plus
an explicit observation delay. ACI is a bounded delayed-feedback variant;
the original paper's guarantees are not asserted for this implementation.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.features.available_weather import utc_times
from gridcast.probabilistic.conformal import KEYS


@dataclass(frozen=True)
class OnlineConfig:
    mode: str = "rolling"  # rolling | adaptive
    window: int = 1440  # most recent eligible finite residuals, not row count
    min_scores: int = 168
    feedback_delay_hours: int = 0
    gamma: float = 0.01
    alpha_min: float = 0.001
    alpha_max: float = 0.5
    window_hours: int | None = None  # optional elapsed feedback-time window

    def __post_init__(self):
        if self.mode not in {"rolling", "adaptive"}:
            raise ValueError("Unknown online calibration mode")
        if (
            not isinstance(self.window, int)
            or not isinstance(self.min_scores, int)
            or not 1 <= self.min_scores <= self.window
        ):
            raise ValueError("Invalid calibration window/minimum")
        if not isinstance(self.feedback_delay_hours, int) or self.feedback_delay_hours < 0:
            raise ValueError("Feedback delay must be nonnegative integer hours")
        if self.window_hours is not None and (
            not isinstance(self.window_hours, int) or self.window_hours < 1
        ):
            raise ValueError("Elapsed window must be positive integer hours")
        if not 0 < self.gamma < 1 or not 0 < self.alpha_min < self.alpha_max < 1:
            raise ValueError("Invalid adaptive settings")


def calibrate_online(
    pred: pd.DataFrame, settings: OnlineConfig, alphas: tuple[float, ...] = tuple(config.INTERVAL_ALPHAS)
) -> pd.DataFrame:
    if not alphas or len(set(alphas)) != len(alphas) or any(not 0 < a < 1 for a in alphas):
        raise ValueError("Invalid nominal miscoverage levels")
    required = {*KEYS, "origin", "target_time", "y", "yhat", "target_imputed", "block"}
    if not required <= set(pred):
        raise ValueError(f"Prediction schema missing {sorted(required - set(pred))}")
    rows = []
    for key, frame in pred.groupby(KEYS, observed=True, sort=True):
        g = frame.sort_values("origin").reset_index(drop=True)
        origins = utc_times(g.origin, "forecast origins")
        targets = utc_times(g.target_time, "targets")
        h = int(g.h.iloc[0])
        if h < 1 or not np.all(targets == origins + pd.Timedelta(hours=h)) or origins.has_duplicates:
            raise ValueError("Invalid forecast key or target alignment")
        feedback = targets + pd.Timedelta(hours=settings.feedback_delay_hours)
        y, f = g.y.to_numpy(float), g.yhat.to_numpy(float)
        valid = np.isfinite(y) & np.isfinite(f) & ~g.target_imputed.to_numpy(bool)
        scores = abs(y - f)
        pending = np.flatnonzero(valid & g.block.isin(["cal", "gap", "test"]).to_numpy())
        pending = pending[np.argsort(feedback.asi8[pending], kind="stable")]
        pool, sorted_scores = deque(), []
        bounds = {}  # (issued row, nominal alpha) -> issued interval, needed for delayed feedback
        effective = {a: a for a in alphas}
        cursor = 0
        last_feedback = pd.NaT
        for i, origin in enumerate(origins):
            while cursor < len(pending) and feedback[pending[cursor]] <= origin:
                j = int(pending[cursor])
                cursor += 1
                insort(sorted_scores, float(scores[j]))
                pool.append((feedback[j], float(scores[j])))
                if len(pool) > settings.window:
                    _, old = pool.popleft()
                    sorted_scores.pop(bisect_left(sorted_scores, old))
                last_feedback = feedback[j]
                if settings.mode == "adaptive" and g.block.iloc[j] == "test":
                    for alpha in alphas:
                        interval = bounds.pop((j, alpha), None)
                        if interval is not None:
                            miss = float(y[j] < interval[0] or y[j] > interval[1])
                            effective[alpha] = float(
                                np.clip(
                                    effective[alpha] + settings.gamma * (alpha - miss),
                                    settings.alpha_min,
                                    settings.alpha_max,
                                )
                            )
            if settings.window_hours is not None:
                cutoff = origin - pd.Timedelta(hours=settings.window_hours)
                while pool and pool[0][0] <= cutoff:
                    _, old = pool.popleft()
                    sorted_scores.pop(bisect_left(sorted_scores, old))
            if g.block.iloc[i] != "test":
                continue
            n = len(sorted_scores)
            for alpha in alphas:
                margin = np.nan
                if n >= settings.min_scores:
                    # Exact order-statistic rank based on actual finite eligible count.
                    rank = min(n, int(np.ceil((n + 1) * (1 - effective[alpha]))))
                    margin = sorted_scores[max(rank - 1, 0)]
                lo, hi = f[i] - margin, f[i] + margin
                if settings.mode == "adaptive" and valid[i] and np.isfinite(margin):
                    bounds[(i, alpha)] = (lo, hi)
                rows.append(
                    {
                        **dict(zip(KEYS, key, strict=True)),
                        "origin": origin,
                        "target_time": targets[i],
                        "feedback_at": feedback[i],
                        "y": y[i],
                        "yhat": f[i],
                        "target_imputed": bool(g.target_imputed.iloc[i]),
                        "alpha": alpha,
                        "effective_alpha": effective[alpha],
                        "lo": lo,
                        "hi": hi,
                        "n_scores": n,
                        "last_feedback_at": last_feedback,
                        "method": f"conformal_{settings.mode}_timestamp",
                    }
                )
    return pd.DataFrame(rows)

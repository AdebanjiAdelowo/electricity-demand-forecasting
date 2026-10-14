"""Canonical forecast tasks and elapsed-day, fold-aware paired evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.features.calendar import season_of

KEY = ["fold", "regime", "h", "origin", "target_time"]


def validate_frame(frame):
    needed = {*KEY, "model", "y", "yhat", "target_imputed"}
    if not needed <= set(frame):
        raise ValueError("Missing prediction schema")
    if frame.duplicated(["model", *KEY]).any():
        raise ValueError("Duplicate model-independent forecast key within model")
    if frame[KEY].isna().any().any() or not frame.regime.eq("none").all():
        raise ValueError("Invalid keys/information regime")
    for column in ["origin", "target_time"]:
        if not isinstance(frame[column].dtype, pd.DatetimeTZDtype) or str(frame[column].dt.tz) != "UTC":
            raise ValueError("Keys require timezone-aware UTC timestamps")
    if (
        not frame.h.isin([1, 24, 48, 168]).all()
        or not (frame.target_time == frame.origin + pd.to_timedelta(frame.h, unit="h")).all()
    ):
        raise ValueError("Invalid horizon/target alignment")
    if not frame.target_imputed.isin([True, False]).all():
        raise ValueError("Invalid imputation flags")


def match(frame, left="ridge", right="lightgbm", schedule=None):
    """Intersection plus full-schedule exclusion accounting; never conceal gaps."""
    validate_frame(frame)
    a = frame[frame.model == left].set_index(KEY).sort_index()
    b = frame[frame.model == right].set_index(KEY).sort_index()
    shared = a.index.intersection(b.index)
    aa, bb = a.loc[shared], b.loc[shared]
    if not np.array_equal(aa.target_imputed, bb.target_imputed) or not np.allclose(
        aa.y, bb.y, rtol=0, atol=0, equal_nan=True
    ):
        raise ValueError("Different observations or imputation flags on shared tasks")
    union = a.index.union(b.index)
    if schedule is not None:
        validate_frame(schedule.assign(model="canonical", yhat=0.0))
        union = pd.MultiIndex.from_frame(schedule[KEY])
        if union.has_duplicates or len(a.index.difference(union)) or len(b.index.difference(union)):
            raise ValueError("Predictions outside canonical schedule")
    if schedule is not None:
        canonical = schedule.set_index(KEY)
        for given in [a, b]:
            expected = canonical.reindex(given.index)
            if not np.array_equal(given.target_imputed, expected.target_imputed) or not np.allclose(
                given.y, expected.y, rtol=0, atol=0, equal_nan=True
            ):
                raise ValueError("Prediction observations differ from canonical schedule")
    supplied_a, supplied_b = a.index, b.index
    # Reindex exposes both missing rows and explicit nonfinite predictions.
    a, b = a.reindex(union), b.reindex(union)
    reasons = []
    for model, g, supplied in [(left, a, supplied_a), (right, b, supplied_b)]:
        present = union.isin(supplied)
        reasons.append(
            pd.DataFrame(
                {
                    **{k: union.get_level_values(k) for k in KEY},
                    "model": model,
                    "missing_key": ~present,
                    "imputed_target": g.target_imputed.eq(True).to_numpy(),
                    "nonfinite": present
                    & (~np.isfinite(g.y.to_numpy(float)) | ~np.isfinite(g.yhat.to_numpy(float))),
                }
            )
        )
    valid = (
        ~a.target_imputed.eq(True)
        & ~b.target_imputed.eq(True)
        & np.isfinite(a.y)
        & np.isfinite(b.y)
        & np.isfinite(a.yhat)
        & np.isfinite(b.yhat)
    )
    paired = a.loc[valid, ["y", "yhat"]].rename(columns={"yhat": "ridge_prediction"})
    paired["lightgbm_prediction"] = b.loc[valid, "yhat"]
    paired["ae_difference"] = abs(paired.y - paired.ridge_prediction) - abs(
        paired.y - paired.lightgbm_prediction
    )
    return paired.reset_index(), pd.concat(reasons, ignore_index=True)


def summary(y, prediction):
    error = np.asarray(prediction, float) - np.asarray(y, float)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Metrics require finite nonempty matched arrays")
    ae = abs(error)
    return {
        "n": len(error),
        "mae_mw": float(ae.mean()),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "signed_mean_error_mw": float(error.mean()),
        "median_absolute_error_mw": float(np.median(ae)),
        **{f"ae_q{int(q * 100)}_mw": float(np.quantile(ae, q)) for q in [0.9, 0.95, 0.99]},
    }


def extreme_threshold(df, start, cutoff, quantile=0.95):
    eligible = df[(df.index >= start) & (df.index < cutoff) & ~df.is_imputed]
    if eligible.empty or not 0 < quantile < 1:
        raise ValueError("Invalid training-only extreme threshold")
    return float(eligible.demand.quantile(quantile))


def score_paired(paired, thresholds):
    if paired.empty:
        raise ValueError("No paired finite evaluation tasks")
    rows = []
    p = paired.copy()
    p["season"] = season_of(pd.DatetimeIndex(p.target_time))
    p["extreme"] = p.y >= p.fold.map(thresholds)
    if p.fold.map(thresholds).isna().any():
        raise ValueError("Missing training threshold")
    for label, columns in [
        ("overall", []),
        ("fold", ["fold"]),
        ("season", ["season"]),
        ("extreme", ["extreme"]),
    ]:
        for key, g in p.groupby(["h", *columns], observed=True):
            key = key if isinstance(key, tuple) else (key,)
            for model, column in [("ridge", "ridge_prediction"), ("lightgbm", "lightgbm_prediction")]:
                rows.append(
                    {
                        "slice": label,
                        "model": model,
                        **dict(zip(["h", *columns], key, strict=True)),
                        **summary(g.y, g[column]),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap(paired, days=(7, 14, 28), replicates=500, seed=0):
    """Sample actual consecutive UTC-day bins, including gaps, within each fold."""
    if replicates < 1 or not days or any(d < 1 for d in days):
        raise ValueError("Invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    rows = []
    for h, g in paired.groupby("h", sort=True):
        daily = (
            g.assign(date=g.origin.dt.floor("D"))
            .groupby(["fold", "date"])
            .ae_difference.agg(["sum", "count"])
        )
        folds = []
        for fold, d in daily.groupby(level="fold", sort=True):
            d = d.droplevel("fold")
            d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D", tz="UTC"), fill_value=0)
            folds.append((fold, d.to_numpy(float)))
        for block in days:
            short = [name for name, a in folds if len(a) < block]
            rec = {
                "h": h,
                "block_days": block,
                "n_paired": len(g),
                "fold_count": len(folds),
                "ridge_minus_lightgbm_mae_mw": float(g.ae_difference.mean()),
                "effective_blocks_approx": sum(len(a) / block for _, a in folds),
                "status": "insufficient_fold_span" if short else "ok",
                "short_folds": ",".join(short),
            }
            if not short:
                totals = np.zeros((replicates, 2))
                for _, a in folds:
                    starts = rng.integers(
                        0, len(a) - block + 1, size=(replicates, int(np.ceil(len(a) / block)))
                    )
                    ix = (starts[..., None] + np.arange(block)).reshape(replicates, -1)[:, : len(a)]
                    totals += a[ix].sum(axis=1)
                valid = totals[:, 1] > 0
                ci = (
                    np.quantile(totals[valid, 0] / totals[valid, 1], [0.025, 0.975])
                    if valid.any()
                    else [np.nan, np.nan]
                )
                rec.update(ci95_low_mw=ci[0], ci95_high_mw=ci[1], valid_replicates=int(valid.sum()))
            rows.append(rec)
    return pd.DataFrame(rows)

"""Export saved forecasts against EIA observations; no fitting or regeneration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from gridcast import config, tracking
from gridcast.data import dataset
from gridcast.evaluation.backtest import scoreable
from gridcast.experiments.revalidate import fingerprint

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluation_rows(pred: pd.DataFrame, observed: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    required = {
        "origin",
        "target_time",
        "h",
        "fold",
        "model",
        "regime",
        "y",
        "yhat",
        "block",
        "target_imputed",
    }
    if not required <= set(pred):
        raise ValueError("Missing prediction columns")
    if pred.origin.dt.tz is None or pred.target_time.dt.tz is None or observed.index.tz is None:
        raise ValueError("Timezone-aware issue/target/observation timestamps required")
    if observed.index.has_duplicates:
        raise ValueError("Duplicate observed demand timestamps")
    if not (pred.target_time == pred.origin + pd.to_timedelta(pred.h, unit="h")).all():
        raise ValueError("Forecast target/issue/horizon misalignment")
    starts = {f["name"]: pd.Timestamp(f["test_start"]) for f in folds}
    ends = {f["name"]: pd.Timestamp(f["test_end"]) for f in folds}
    if not set(pred.fold) <= set(ends):
        raise ValueError("Unknown fold")
    p = scoreable(pred).copy()
    p = p[
        (p.origin >= p.fold.map(starts)) & (p.origin < p.fold.map(ends)) & (p.target_time < p.fold.map(ends))
    ]
    if p.duplicated(["model", "regime", "fold", "origin", "h"]).any():
        raise ValueError("Duplicate forecast keys")
    actual = observed.reindex(pd.DatetimeIndex(p.target_time))
    if actual.demand.isna().any() or actual.is_imputed.any():
        raise ValueError("Missing/imputed observations in scored rows")
    np.testing.assert_allclose(p.y, actual.demand.to_numpy(), rtol=0, atol=1e-7)
    if "demand_raw" in actual:
        np.testing.assert_allclose(p.y, actual.demand_raw.to_numpy(), rtol=0, atol=1e-7)
    if not np.isfinite(p[["y", "yhat"]].to_numpy()).all():
        raise ValueError("Nonfinite forecast/observation")
    out = p[["origin", "target_time", "h", "fold", "model", "y", "yhat"]].rename(
        columns={
            "origin": "issue_time_utc",
            "target_time": "target_time_utc",
            "h": "horizon_hours",
            "y": "observed_demand_mw",
            "yhat": "predicted_demand_mw",
        }
    )
    out["signed_error_mw"] = out.predicted_demand_mw - out.observed_demand_mw
    out["absolute_error_mw"] = abs(out.signed_error_mw)
    return out.sort_values(["fold", "model", "horizon_hours", "target_time_utc"]).reset_index(drop=True)


def reproduce_metrics(rows: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    result = []
    for (model, h), g in rows.groupby(["model", "horizon_hours"]):
        ref = published[(published.model == model) & (published.h == h) & (published.regime == "none")]
        if len(ref) != 1:
            raise ValueError("Missing or ambiguous published metric")
        r = ref.iloc[0]
        mae = float(g.absolute_error_mw.mean())
        rmse = float(np.sqrt(np.mean(g.signed_error_mw**2)))
        if len(g) != r.n:
            raise ValueError("Published row count differs")
        np.testing.assert_allclose([mae, rmse], [r.mae, r.rmse], rtol=1e-12, atol=1e-8)
        result.append(
            {
                "model": model,
                "horizon_hours": h,
                "n": len(g),
                "mae_mw": mae,
                "rmse_mw": rmse,
                "mae_difference_from_published_mw": mae - r.mae,
                "rmse_difference_from_published_mw": rmse - r.rmse,
            }
        )
    return pd.DataFrame(result)


def select_weeks(
    rows: pd.DataFrame, folds: list[dict], models: list[str], horizons: list[int]
) -> pd.DataFrame:
    windows = []
    for fold in folds:
        start = pd.Timestamp(fold["test_start"]) + pd.Timedelta(hours=max(horizons))
        end = pd.Timestamp(fold["test_end"])
        while start + pd.Timedelta(days=7) <= end:
            stop = start + pd.Timedelta(days=7)
            p = rows[
                (rows.fold == fold["name"])
                & (rows.target_time_utc >= start)
                & (rows.target_time_utc < stop)
                & rows.model.isin(models)
                & rows.horizon_hours.isin(horizons)
            ]
            counts = p.groupby(["model", "horizon_hours"]).size()
            if len(counts) == len(models) * len(horizons) and (counts == 168).all():
                reference = p[(p.model == models[0]) & (p.horizon_hours == 24)]
                if len(reference) != 168:
                    raise ValueError("Week selection requires the 24-hour horizon")
                windows.append(
                    {
                        "fold": fold["name"],
                        "start": start,
                        "end": stop,
                        "selection_model": models[0],
                        "horizon_hours": 24,
                        "selection_mae_mw": reference.absolute_error_mw.mean(),
                    }
                )
            start = stop
    if len(windows) < 3:
        raise ValueError("Need three complete seven-day windows")
    pool = pd.DataFrame(windows).sort_values(["selection_mae_mw", "start"]).reset_index(drop=True)
    selection = pool.iloc[[0, len(pool) // 2, -1]].copy()
    selection["example"] = ["good", "typical", "poor"]
    selection["eligible_windows"] = len(pool)
    return selection


def plot_week(rows: pd.DataFrame, window, out: Path):
    p = rows[rows.horizon_hours == 24]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    actual = p.drop_duplicates("target_time_utc").sort_values("target_time_utc")
    axes[0].plot(
        actual.target_time_utc, actual.observed_demand_mw, color="black", label="Observed EIA demand"
    )
    for name, g in p.groupby("model"):
        g = g.sort_values("target_time_utc")
        axes[0].plot(g.target_time_utc, g.predicted_demand_mw, label=f"{name}, 24h forecast", alpha=0.8)
        axes[1].plot(g.target_time_utc, g.signed_error_mw, label=name)
    axes[0].set(
        ylabel="Demand (MW)",
        title=f"{window.example.title()} example: {window.fold}; selected by {window.selection_model} 24h MAE",
    )
    axes[1].axhline(0, color="black", ls=":")
    axes[1].set(ylabel="Predicted − observed (MW)", xlabel="Target hour ending (UTC)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / f"actual_vs_predicted_{window.example}.png", dpi=160)
    fig.savefig(out / f"actual_vs_predicted_{window.example}.svg")
    plt.close(fig)


def run(source: Path, output: Path, models: list[str], horizons: list[int]) -> None:
    if output.exists():
        raise ValueError("Use a new output directory; never overwrite evidence")
    if (
        "ridge" not in models
        or 24 not in horizons
        or len(set(models)) != len(models)
        or len(set(horizons)) != len(horizons)
    ):
        raise ValueError("Unique models/horizons, including ridge and 24h, required")
    models = ["ridge", *[name for name in models if name != "ridge"]]
    record = json.loads((source / "run.json").read_text())
    if fingerprint(dataset.PROCESSED) != record["data"]["processed_sha256"]:
        raise ValueError("Dataset fingerprint differs from saved run")
    files = sorted(source.glob("*.parquet"))
    hashes = {p.name: fingerprint(p) for p in files}
    predictions = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    if not set(models) <= set(predictions.model) or not set(horizons) <= set(predictions.h):
        raise ValueError("Requested model/horizon unavailable")
    if set(predictions.regime) != {"none"}:
        raise ValueError("This export validates the primary no-weather benchmark")
    rows = evaluation_rows(predictions, dataset.load(), record["folds"])
    scores = reproduce_metrics(rows, pd.read_csv(source / "overall.csv"))
    windows = select_weeks(rows, record["folds"], models, horizons)
    output.mkdir(parents=True)
    scores.to_csv(output / "independently_reproduced_metrics.csv", index=False)
    windows.to_csv(output / "selected_periods.csv", index=False)
    counts = {}
    for window in windows.itertuples():
        p = rows[
            (rows.fold == window.fold)
            & rows.model.isin(models)
            & rows.horizon_hours.isin(horizons)
            & (rows.target_time_utc >= window.start)
            & (rows.target_time_utc < window.end)
        ]
        path = output / f"predictions_vs_actual_{window.example}.csv"
        p.to_csv(path, index=False)
        reread = pd.read_csv(path)
        np.testing.assert_allclose(reread.signed_error_mw, p.signed_error_mw, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(abs(reread.signed_error_mw), reread.absolute_error_mw, atol=1e-8)
        counts[window.example] = len(p)
        plot_week(p, window, output)
    for p in files:
        if fingerprint(p) != hashes[p.name]:
            raise RuntimeError("Original prediction file changed")
    manifest = {
        "source_run": str(source.relative_to(config.ROOT)),
        "source_commit": record["git"],
        "export_commit": tracking.git_state(),
        "dataset_sha256": fingerprint(dataset.PROCESSED),
        "prediction_sha256": hashes,
        "models": models,
        "horizons": horizons,
        "csv_row_counts": counts,
        "full_benchmark_metric_rows_verified": len(scores),
        "units": "MW",
        "signed_error": "predicted minus observed",
        "selection": "Complete 168-hour blocks anchored at fold start plus maximum horizon; minimum, median-ranked and maximum Ridge 24h MAE. Post-hoc illustrations, not unbiased aggregate estimates.",
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": fingerprint(p)} for p in output.iterdir()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "README.md").write_text(
        "# Saved predictions versus EIA demand\n\n"
        "These CSVs contain original saved point forecasts and non-imputed observed EIA demand. All times are UTC hour-ending labels; issue time assumes current-hour demand is already known. Signed error is predicted minus observed, in MW.\n\n"
        "The good, typical and poor seven-day windows are selected after evaluation by Ridge 24-hour MAE among complete common windows. The figures show the 24-hour horizon; CSVs include all requested horizons. This illustrates variation and is not an unbiased measure of aggregate skill.\n\n"
        "Full-data MAE and RMSE for all five models and four horizons were independently reproduced before selecting examples. See independently_reproduced_metrics.csv, selected_periods.csv and manifest.json. No model was refitted and no original artifact was overwritten.\n"
    )
    print(f"Verified {len(scores)} full-benchmark metric rows; exported {counts}; {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=config.RESULTS / "forecasting_baselines/20261012T093203Z")
    ap.add_argument(
        "--output",
        type=Path,
        default=config.RESULTS / "prediction_exports" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ"),
    )
    ap.add_argument("--models", nargs="+", default=["ridge", "lightgbm"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 24, 48, 168])
    args = ap.parse_args()
    run(args.source, args.output, args.models, args.horizons)

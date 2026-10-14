"""Preparation defaults to a dry run; historical training needs explicit gates."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import platform
import resource
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from gridcast import config, tracking
from gridcast.data import dataset
from gridcast.evaluation import fair_comparison as evaluation
from gridcast.experiments.revalidate import fingerprint
from gridcast.features.build import FeatureSpec, make_features, one_hot_design
from gridcast.models.base import train_mask

DEFAULT = config.ROOT / "configs/fair_model_comparison.json"


class ResourceLimit(RuntimeError):
    pass


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


@contextmanager
def budget(seconds, memory_mib):
    start = time.perf_counter()
    previous = signal.getsignal(signal.SIGALRM)

    def check(*_):
        if peak_rss() > memory_mib * 1024**2 or time.perf_counter() - start > seconds:
            raise ResourceLimit(
                "CPU time or peak-memory limit exceeded; stop, retain evidence and request approval"
            )

    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, 1, 1)
    try:
        check()
        yield check
        check()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def validate_settings(settings):
    if (
        settings["schema_version"] != 1
        or settings["threads"] not in [1, 2]
        or settings["horizons"] != [1, 24, 48, 168]
    ):
        raise ValueError("Unsupported schema, threads or horizon protocol")
    if settings["feedback_delay_hours"] != 0:
        raise ValueError("This protocol fixes zero-delay retrospective target availability")
    blocks = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in settings["development_blocks"]]
    start = pd.Timestamp(settings["train_start"])
    if (
        start.tz is None
        or not blocks
        or any(a.tz is None or b.tz is None or not start < a < b for a, b in blocks)
    ):
        raise ValueError("Invalid chronological development blocks")
    if any(blocks[i][1] > blocks[i + 1][0] for i in range(len(blocks) - 1)) or blocks[-1][1] > pd.Timestamp(
        "2024-04-01", tz="UTC"
    ):
        raise ValueError("Overlapping development blocks or primary-period access")
    if not 0 < settings["extreme_training_quantile"] < 1 or not 1 <= settings["memory_mib"] <= 2048:
        raise ValueError("Invalid threshold or memory cap")
    if (
        not 0 < settings["preparation_seconds"] <= 300
        or not 0 < settings["campaign_seconds"] <= 7200
        or not 0 < settings["model_search_seconds"] <= settings["campaign_seconds"]
    ):
        raise ValueError("Invalid bounded resource protocol")
    return blocks


def candidates(settings):
    """Fixed ordered enumeration; no random search or performance-driven expansion."""
    ridge = [
        {"id": f"r{i:02d}", "model": "ridge", "parameters": {"alpha": float(a)}}
        for i, a in enumerate(settings["ridge_alphas"])
    ]
    light = []
    for profile in settings["lightgbm_profiles"]:
        if set(profile) != {"learning_rate", "n_estimators", "min_child_samples", "reg_alpha", "reg_lambda"}:
            raise ValueError("Unknown model-profile parameter; CPU/thread safeguards cannot be overridden")
        for objective in settings["lightgbm_objectives"]:
            for leaves in settings["lightgbm_leaves"]:
                light.append(
                    {
                        "id": f"l{len(light):02d}",
                        "model": "lightgbm",
                        "parameters": {**profile, "num_leaves": leaves, "objective": objective},
                    }
                )
    if (
        not ridge
        or not light
        or len(ridge) > 8
        or len(light) > 12
        or any(c["parameters"]["alpha"] <= 0 for c in ridge)
    ):
        raise ValueError("Invalid or oversized tuning grid")
    for c in light:
        p = c["parameters"]
        if (
            p["objective"] not in ["l1", "l2"]
            or not 1 <= p["n_estimators"] <= 600
            or not 2 <= p["num_leaves"] <= 63
            or not 0 < p["learning_rate"] <= 0.2
            or p["min_child_samples"] < 1
            or min(p["reg_alpha"], p["reg_lambda"]) < 0
        ):
            raise ValueError("Invalid LightGBM candidate")
    return {"ridge": ridge, "lightgbm": light}


def shared_design(features):
    """Fixed one-hot basis retains month as well as existing hour/offday design."""
    x = one_hot_design(features)
    for month in range(1, 13):
        x[f"oh_month{month}"] = (features.c_month == month).astype(float)
    if any(c.startswith("w_") for c in x):
        raise ValueError("Weather prohibited in equal-information comparison")
    return x.replace([np.inf, -np.inf], np.nan)


def prepare(df, h):
    if df.index.tz is None or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("Demand requires ordered unique UTC hourly data")
    if str(df.index.tz) != "UTC" or not df.index.as_unit("ns").equals(
        pd.date_range(df.index[0], df.index[-1], freq="h", tz="UTC").as_unit("ns")
    ):
        raise ValueError("Incomplete hourly grid")
    x, y, m = make_features(df, h, FeatureSpec(regime="none"))
    imputed = df.is_imputed.astype("boolean").shift(-h, fill_value=True).astype(bool)
    return shared_design(x), y, m, imputed


def fitting_mask(x, y, m, imputed, h, start, cutoff):
    return train_mask(x.index, h, start, cutoff) & np.isfinite(y) & np.isfinite(m) & (m > 0) & ~imputed


def schedule(df, h, fold, start, end):
    origins = df.index[(df.index >= start) & (df.index < end) & (df.index + pd.Timedelta(hours=h) < end)]
    target = origins + pd.Timedelta(hours=h)
    return pd.DataFrame(
        {
            "fold": fold,
            "regime": "none",
            "h": h,
            "origin": origins,
            "target_time": target,
            "y": df.demand.reindex(target).to_numpy(),
            "target_imputed": df.is_imputed.reindex(target).to_numpy(bool),
        }
    )


def estimator(candidate, settings):
    if candidate["model"] == "ridge":
        model = Ridge(**candidate["parameters"])
    else:
        model = lgb.LGBMRegressor(
            **candidate["parameters"],
            random_state=settings["seed"],
            n_jobs=settings["threads"],
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
            device_type="cpu",
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
        )
    # The same learned input transformations and identical columns for both.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def fit_predict(df, h, start, cutoff, tasks, candidate, settings, event_sink=None):
    feature_started = time.perf_counter()
    x, y, m, imputed = prepare(df, h)
    feature_seconds = time.perf_counter() - feature_started
    train = fitting_mask(x, y, m, imputed, h, start, cutoff)
    if not train.any() or x.loc[train].notna().sum().eq(0).any():
        raise ValueError("Empty fitting block or wholly missing feature")
    model = estimator(candidate, settings)
    t = time.perf_counter()
    event = {
        "model": candidate["model"],
        "candidate_id": candidate["id"],
        "h": h,
        "cutoff": str(cutoff),
        "train_n": int(train.sum()),
    }
    if event_sink:
        event_sink({**event, "event": "fit_started"})
    try:
        model.fit(x.loc[train], y.loc[train])
    except Exception as exc:
        if event_sink:
            event_sink({**event, "event": "fit_failed", "error": str(exc)})
        raise
    if event_sink:
        event_sink({**event, "event": "fit_completed", "seconds": time.perf_counter() - t})
    fitting = time.perf_counter() - t
    t = time.perf_counter()
    pred = tasks.copy()
    pred["model"] = candidate["model"]
    pred["yhat"] = model.predict(x.reindex(tasks.origin)) * m.reindex(tasks.origin).to_numpy()
    if not np.isfinite(pred.yhat).all():
        raise ValueError("Incomplete finite prediction schedule")
    record = {
        "fit_seconds": fitting,
        "feature_preparation_seconds": feature_seconds,
        "n_features": len(x.columns),
        "predict_seconds": time.perf_counter() - t,
        "train_n": int(train.sum()),
        "latest_fit_target": str((x.index[train] + pd.Timedelta(hours=h)).max()),
        "fit_cutoff": str(cutoff),
        "feature_names": list(x.columns),
        "peak_rss_bytes": peak_rss(),
        "preprocessing_fit_n": int(model.named_steps["standardscaler"].n_samples_seen_),
        "iterations": int(getattr(model.steps[-1][1], "n_estimators_", 1)),
        "validation_schedule_n": len(tasks),
        "validation_eligible_n": int((~tasks.target_imputed).sum()),
        "early_stopping": False,
    }
    return pred, model, record


def select(table):
    """Require all blocks/keys; minimize pooled MW MAE then fixed candidate ID."""
    rows = []
    for (model, h), g in table.groupby(["model", "h"], sort=True):
        valid = g[g.complete & np.isfinite(g.mae_mw)]
        if valid.empty:
            raise ValueError(f"No complete candidate for {model}/{h}; all failures retained")
        winner = valid.sort_values(["mae_mw", "candidate_id"], kind="stable").iloc[0]
        rows.append(
            {
                "model": model,
                "h": int(h),
                "candidate_id": winner.candidate_id,
                "mae_mw": float(winner.mae_mw),
                "validation_n": int(winner.n),
            }
        )
    return rows


def fit_ledger(out):
    def save(event):
        with (out / "fit_events.jsonl").open("a") as handle:
            handle.write(json.dumps({"utc": str(pd.Timestamp.now(tz="UTC")), **event}) + "\n")

    return save


def search(df, settings, grid, out, check):
    blocks = validate_settings(settings)
    df = df.loc[df.index < blocks[-1][1]].copy()
    frames = []
    table = []
    fits = []
    start = pd.Timestamp(settings["train_start"])
    for name, choices in grid.items():
        model_started = time.perf_counter()
        for h in settings["horizons"]:
            for candidate in choices:
                pieces = []
                complete = True
                failure = ""
                for i, (a, b) in enumerate(blocks):
                    check()
                    if time.perf_counter() - model_started > settings["model_search_seconds"]:
                        raise ResourceLimit(f"{name} search reached its common model budget")
                    tasks = schedule(df, h, f"dev{i + 1}", a, b)
                    try:
                        pred, model, record = fit_predict(
                            df, h, start, a, tasks, candidate, settings, event_sink=fit_ledger(out)
                        )
                        evaluation.validate_frame(pred)
                        if len(pred) != len(tasks) or not pred[evaluation.KEY].equals(tasks[evaluation.KEY]):
                            raise ValueError("Candidate forecast keys differ from canonical schedule")
                        if (
                            not np.array_equal(pred.target_imputed, tasks.target_imputed)
                            or not np.allclose(pred.y, tasks.y, rtol=0, atol=0, equal_nan=True)
                            or not np.isfinite(pred.yhat).all()
                        ):
                            raise ValueError(
                                "Candidate changed observations, exclusions or finite completeness"
                            )
                        eligible_block = pred[~pred.target_imputed]
                        record["validation_mae_mw"] = evaluation.summary(
                            eligible_block.y, eligible_block.yhat
                        )["mae_mw"]
                        record["validation_rmse_mw"] = evaluation.summary(
                            eligible_block.y, eligible_block.yhat
                        )["rmse_mw"]
                        pieces.append(pred.assign(candidate_id=candidate["id"]))
                        fits.append(
                            {
                                "model": name,
                                "h": h,
                                "candidate_id": candidate["id"],
                                "block": f"dev{i + 1}",
                                **record,
                            }
                        )
                        del model
                        gc.collect()
                    except ResourceLimit:
                        raise
                    except Exception as exc:
                        complete = False
                        failure = f"dev{i + 1}: {type(exc).__name__}: {exc}"
                        fits.append(
                            {
                                "model": name,
                                "h": h,
                                "candidate_id": candidate["id"],
                                "block": f"dev{i + 1}",
                                "failed": True,
                                "error": failure,
                                "peak_rss_bytes": peak_rss(),
                            }
                        )
                        break
                joined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
                eligible = (
                    joined[~joined.target_imputed & np.isfinite(joined.y) & np.isfinite(joined.yhat)]
                    if len(joined)
                    else joined
                )
                rec = (
                    evaluation.summary(eligible.y, eligible.yhat)
                    if len(eligible)
                    else {"n": 0, "mae_mw": np.nan, "rmse_mw": np.nan}
                )
                table.append(
                    {
                        "model": name,
                        "h": h,
                        "candidate_id": candidate["id"],
                        "complete": complete,
                        "failure": failure,
                        "blocks_completed": len(pieces),
                        **rec,
                    }
                )
                if len(joined):
                    frames.append(joined)
                pd.DataFrame(table).to_csv(out / "candidate_scores.csv", index=False)
                pd.DataFrame(fits).to_csv(out / "fits.csv", index=False)
                if time.perf_counter() - model_started > settings["model_search_seconds"]:
                    raise ResourceLimit(f"{name} search reached its common model budget")
    table = pd.DataFrame(table)
    chosen = select(table)
    prediction = pd.concat(frames, ignore_index=True)
    prediction.to_parquet(out / "development_candidate_predictions.parquet", index=False)
    frozen = {
        "policies": chosen,
        "selection": "pooled MW MAE; exact ties fixed candidate ID",
        "configuration_sha256": fingerprint(out / "configuration.json"),
        "validation_predictions_sha256": fingerprint(out / "development_candidate_predictions.parquet"),
        "frozen_at_utc": str(pd.Timestamp.now(tz="UTC")),
    }
    (out / "selection.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return chosen, fits


def evaluate(df, settings, grid, chosen, folds, out, check, synthetic=False):
    frozen = fingerprint(out / "selection.json")
    predictions = []
    thresholds = {}
    all_tasks = []
    fit_records = []
    for fold, start, cutoff, a, b in folds:
        thresholds[fold] = evaluation.extreme_threshold(
            df, start, cutoff, settings["extreme_training_quantile"]
        )
        for h in settings["horizons"]:
            tasks = schedule(df, h, fold, a, b)
            all_tasks.append(tasks)
            for name in ["ridge", "lightgbm"]:
                check()
                policy = next(p for p in chosen if p["model"] == name and p["h"] == h)
                candidate = next(c for c in grid[name] if c["id"] == policy["candidate_id"])
                pred, model, record = fit_predict(
                    df, h, start, cutoff, tasks, candidate, settings, event_sink=fit_ledger(out)
                )
                predictions.append(pred)
                fit_records.append({"fold": fold, "h": h, "model": name, **record})
                with (out / f"{name}_{fold}_h{h}.pkl").open("wb") as handle:
                    pickle.dump(model, handle)
                del model
                gc.collect()
    frame = pd.concat(predictions, ignore_index=True)
    frame.to_parquet(out / "evaluation_predictions.parquet", index=False)
    paired, missing = evaluation.match(frame, schedule=pd.concat(all_tasks, ignore_index=True))
    missing.to_csv(out / "key_accounting.csv", index=False)
    evaluation.score_paired(paired, thresholds).to_csv(out / "paired_metrics.csv", index=False)
    evaluation.bootstrap(
        paired, tuple(settings["bootstrap_days"]), settings["bootstrap_replicates"], settings["seed"]
    ).to_csv(out / "paired_bootstrap.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
    pd.DataFrame(fit_records).to_csv(out / "evaluation_fits.csv", index=False)
    if not synthetic:
        # Unchanged reference predictions are not used for selection or tuning.
        baseline = config.RESULTS / "forecasting_baselines/20261012T093203Z"
        refs = []
        for name in ["ridge", "lightgbm"]:
            p = pd.read_parquet(baseline / f"{name}.parquet")
            end = {f.name: f.test_end for f in config.TEST_FOLDS}
            p = p[(p.block == "test") & (p.target_time < p.fold.map(end))].copy()
            p["model"] = "original_" + name
            refs.append(p)
        reference = pd.concat(refs)
        reference.to_parquet(out / "original_reference_predictions.parquet", index=False)
        ref_pair, ref_missing = evaluation.match(
            reference, "original_ridge", "original_lightgbm", pd.concat(all_tasks, ignore_index=True)
        )
        evaluation.score_paired(ref_pair, thresholds).to_csv(
            out / "original_reference_metrics.csv", index=False
        )
        ref_missing.to_csv(out / "original_reference_key_accounting.csv", index=False)
    if frozen != fingerprint(out / "selection.json"):
        raise ValueError("Frozen selection changed during evaluation")


def load_input():
    # Read only the two permitted data columns; Arrow decoding stays serial.
    return pd.read_parquet(dataset.PROCESSED, columns=["demand", "is_imputed"], use_threads=False)


def synthetic_data():
    rng = np.random.default_rng(0)
    ix = pd.date_range("2022-01-01", periods=10000, freq="h", tz="UTC")
    t = np.arange(len(ix))
    return pd.DataFrame(
        {
            "demand": 40000
            + 5000 * np.sin(2 * np.pi * t / 24)
            + 2000 * np.cos(2 * np.pi * t / 168)
            + rng.normal(0, 100, len(ix)),
            "is_imputed": False,
        },
        index=ix,
    )


def plan(settings):
    validate_settings(settings)
    g = candidates(settings)
    return {
        "status": "PREPARATION_ONLY: full tuning/evaluation NOT executed",
        "grid": g,
        "search_fits": {
            m: len(c) * len(settings["horizons"]) * len(settings["development_blocks"]) for m, c in g.items()
        },
        "final_primary_refits": 32,
        "early_stopping": False,
        "primary_test_status": "previously inspected, exploratory",
        "configuration": settings,
    }


def authorize(mode, historical=False, primary=False):
    if mode == "development" and not historical or mode == "full" and not (historical and primary):
        raise PermissionError(
            "Historical tuning/full primary evaluation requires separate explicit authorization flags"
        )
    if mode not in ["dry-run", "smoke", "development", "full"]:
        raise ValueError("Unknown execution mode")


def run(settings, mode="dry-run", historical=False, primary=False, output=None):
    authorize(mode, historical, primary)
    spec = plan(settings)
    if mode == "dry-run":
        print(json.dumps(spec, indent=2))
        return None
    started = time.perf_counter()
    original = settings
    settings = json.loads(json.dumps(settings))
    g = candidates(settings)
    is_smoke = mode == "smoke"
    if is_smoke:
        df = synthetic_data()
        settings["train_start"] = str(df.index[336])
        settings["development_blocks"] = [
            [str(df.index[a]), str(df.index[b])] for a, b in [(3600, 4800), (4800, 6000), (6000, 7200)]
        ]
        g = {
            "ridge": [g["ridge"][1]],
            "lightgbm": [
                {
                    **g["lightgbm"][0],
                    "parameters": {**g["lightgbm"][0]["parameters"], "n_estimators": 20, "num_leaves": 15},
                }
            ],
        }
    else:
        df = load_input()
        expected = json.loads(
            (config.RESULTS / "forecasting_baselines/20261012T093203Z/run.json").read_text()
        )["data"]["processed_sha256"]
        if fingerprint(dataset.PROCESSED) != expected:
            raise ValueError("Dataset changed since original benchmark")
        if mode == "development":
            df = df[df.index < pd.Timestamp("2024-04-01", tz="UTC")]
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    out = output or config.RESULTS / "fair_model_comparison" / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "configuration.json").write_text(
        json.dumps({"requested": original, "executed": settings, "executed_grid": g, "mode": mode}, indent=2)
        + "\n"
    )
    inputs = {
        str(p.relative_to(config.ROOT)): fingerprint(p)
        for p in [dataset.PROCESSED, dataset.MANIFEST]
        if p.exists()
    }
    metadata = {
        "git": tracking.git_state(),
        "python": sys.version,
        "versions": tracking.versions(),
        "hardware": {"platform": platform.platform(), "machine": platform.machine()},
        "device": "cpu",
        "threads": settings["threads"],
        "thread_environment": {
            k: os.environ.get(k) for k in ["GRIDCAST_JOBS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
        "lock_sha256": fingerprint(config.ROOT / "uv.lock"),
        "input_sha256": inputs,
        "seed": settings["seed"],
        "data_used": "synthetic fixture only" if is_smoke else "fixed EIA snapshot",
        "source_sha256": {
            str(p.relative_to(config.ROOT)): fingerprint(p) for p in (config.ROOT / "src").rglob("*.py")
        },
        "stage": mode,
        "full_benchmark_executed": mode == "full",
        "actual_grid": g,
        "key": evaluation.KEY,
        "primary_folds": [tracking.fold_dict(f) for f in config.TEST_FOLDS],
    }
    status = "failed"
    try:
        with (
            budget(
                settings["preparation_seconds"] if is_smoke else settings["campaign_seconds"],
                settings["memory_mib"],
            ) as check,
            threadpool_limits(limits=settings["threads"]),
        ):
            chosen, fits = search(df, settings, g, out, check)
            if is_smoke or mode == "full":
                folds = (
                    [
                        (
                            "synthetic_holdout",
                            pd.Timestamp(settings["train_start"]),
                            df.index[7200],
                            df.index[7200],
                            df.index[-1] + pd.Timedelta(hours=1),
                        )
                    ]
                    if is_smoke
                    else [
                        (f.name, f.train_start, f.cal_start, f.test_start, f.test_end)
                        for f in config.TEST_FOLDS
                    ]
                )
                evaluate(df, settings, g, chosen, folds, out, check, is_smoke)
            metadata["search_attempt_count"] = len(fits)
            metadata["search_completed_fit_count"] = sum(not r.get("failed", False) for r in fits)
            metadata["search_failed_attempt_count"] = sum(bool(r.get("failed", False)) for r in fits)
            metadata["configuration_sha256"] = fingerprint(out / "configuration.json")
            metadata["dataset_manifest"] = tracking.data_manifest()
            metadata["evaluation_completed_fit_count"] = (
                len(pd.read_csv(out / "evaluation_fits.csv")) if (out / "evaluation_fits.csv").exists() else 0
            )
            metadata["selection"] = chosen
            for path, expected in inputs.items():
                if fingerprint(config.ROOT / path) != expected:
                    raise ValueError("Input mutated during experiment")
        status = "completed"
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if (out / "fit_events.jsonl").exists():
            events = [json.loads(line) for line in (out / "fit_events.jsonl").read_text().splitlines()]
            metadata["actual_fit_started_count"] = sum(e["event"] == "fit_started" for e in events)
            metadata["actual_fit_completed_count"] = sum(e["event"] == "fit_completed" for e in events)
            metadata["actual_fit_failed_count"] = sum(e["event"] == "fit_failed" for e in events)
        metadata.update(status=status, wall_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss())
        (out / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--mode", choices=["dry-run", "smoke", "development", "full"], default="dry-run")
    parser.add_argument("--authorize-historical-search", action="store_true")
    parser.add_argument("--authorize-primary-evaluation", action="store_true")
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.mode,
        args.authorize_historical_search,
        args.authorize_primary_evaluation,
    )

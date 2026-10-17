"""Six fixed real-data fits for resources only; never select or score models."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gridcast import config, resources, tracking
from gridcast.data import dataset, snapshot
from gridcast.evaluation.fair_comparison import KEY, validate_frame
from gridcast.experiments import fair_model_comparison as fair
from gridcast.experiments.revalidate import fingerprint

CUTOFF = pd.Timestamp("2023-07-01", tz="UTC")
END = pd.Timestamp("2023-07-15", tz="UTC")
HORIZONS = (1, 168)
SECONDS = 300
MEMORY_MIB = 2048
EARLY_STOP_MIB = int(0.9 * MEMORY_MIB)
IDS = {"ridge": ("r01",), "lightgbm": ("l02", "l11")}


class PressureBlocked(RuntimeError):
    pass


def pressure():
    if platform.system() == "Linux":
        return {"utc": str(pd.Timestamp.now(tz="UTC")), **resources.linux_snapshot(disk_path=config.RESULTS)}
    if platform.system() != "Darwin":
        raise PressureBlocked("No verified system-pressure admission gate on this platform")
    level = int(
        subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True, timeout=5
        ).strip()
    )
    # This sysctl returns dispatch flags, not the internal XNU enum.
    return {
        "utc": str(pd.Timestamp.now(tz="UTC")),
        "dispatch_pressure_level": level,
        "meaning": {1: "normal", 2: "warning", 4: "critical"}.get(level, "unknown"),
    }


def admit(record, threads=resources.MAX_THREADS):
    if record.get("sensor") == "linux":
        try:
            resources.linux_admit(record, MEMORY_MIB, threads)
        except resources.AdmissionBlocked as exc:
            raise PressureBlocked(f"Linux resource admission blocked: {exc}") from exc
        return
    if record["dispatch_pressure_level"] != 1:
        raise PressureBlocked(
            f"Memory pressure {record['meaning']} ({record['dispatch_pressure_level']}); no fit authorized under pressure"
        )


def pilot_candidates(settings):
    grid = fair.candidates(settings)
    return [
        c
        for name, identifiers in IDS.items()
        for identifier in identifiers
        for c in grid[name]
        if c["id"] == identifier
    ]


def load_prefix():
    df = pd.read_parquet(
        dataset.PROCESSED,
        columns=["demand", "is_imputed"],
        use_threads=False,
        filters=[("utc_end", "<", END.to_pydatetime())],
    )
    if df.empty or not (df.index < END).all() or str(df.index.tz) != "UTC":
        raise ValueError("Pilot reader exposed rows outside its pre-test prefix")
    return df


def plan(settings):
    fair.validate_settings(settings)
    return {
        "purpose": "RESOURCE ONLY; no MAE selection, full search or primary evaluation",
        "fit_start": settings["train_start"],
        "target_cutoff": str(CUTOFF),
        "validation_end": str(END),
        "horizons": HORIZONS,
        "candidates": pilot_candidates(settings),
        "planned_fits": 6,
        "seconds_cap": SECONDS,
        "nominal_memory_mib": MEMORY_MIB,
        "early_stop_mib": EARLY_STOP_MIB,
        "threads": settings["threads"],
        "config_unchanged": True,
    }


def run(settings, execute=False, output=None, preflight_only=False):
    if execute and preflight_only:
        raise ValueError("Select either pilot execution or preflight")
    protocol = plan(settings)
    if not execute and not preflight_only:
        print(json.dumps(protocol, indent=2))
        return None
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    out = output or config.RESULTS / "fair_model_comparison" / f"resource_pilot_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "pilot_plan.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "configuration.json").write_text(json.dumps(settings, indent=2) + "\n")
    started = time.perf_counter()
    completed = []
    pressure_records = []
    record = {
        "git": tracking.git_state(),
        "stage": "resource_pilot",
        "readiness": "BLOCKED",
        "status": "not_started",
        "device": "cpu",
        "threads": settings["threads"],
        "versions": tracking.versions(),
        "platform": platform.platform(),
        "planned_fits": 6,
        "completed_fits": 0,
        "full_search_executed": False,
        "primary_evaluation_executed": False,
        "selection_executed": False,
        "lock_sha256": fingerprint(config.ROOT / "uv.lock"),
        "scientific_configuration_sha256": fingerprint(out / "configuration.json"),
        "source_sha256": {
            str(p.relative_to(config.ROOT)): fingerprint(p) for p in (config.ROOT / "src").rglob("*.py")
        },
        "disk_free_bytes": shutil.disk_usage(config.ROOT).free,
        "pressure_observations": pressure_records,
    }
    try:
        first = pressure()
        pressure_records.append(first)
        admit(first, settings["threads"])
        if preflight_only:
            record["status"] = "preflight_normal_no_fits"
            record["readiness"] = "CONDITIONAL_NO_MEASUREMENTS"
            print(out)
            return out
        with fair.budget(SECONDS, EARLY_STOP_MIB) as check, threadpool_limits(limits=settings["threads"]):
            expected = snapshot.expected_processed_sha256()
            observed = fingerprint(dataset.PROCESSED)
            if expected != observed:
                raise ValueError("Input differs from immutable benchmark")
            record["processed_sha256"] = observed
            record["manifest_sha256"] = fingerprint(dataset.MANIFEST)
            load_started = time.perf_counter()
            df = load_prefix()
            record["prefix_load_seconds"] = time.perf_counter() - load_started
            record["returned_data_start"] = str(df.index.min())
            record["returned_data_end"] = str(df.index.max())
            start = pd.Timestamp(settings["train_start"])
            for h in HORIZONS:
                tasks = fair.schedule(df, h, "resource_dev1", CUTOFF, END)
                for candidate in pilot_candidates(settings):
                    current = pressure()
                    pressure_records.append(current)
                    admit(current, settings["threads"])
                    check()
                    attempt = f"{candidate['id']}_h{h}"
                    record["active_attempt"] = attempt
                    (out / "active_attempt.json").write_text(
                        json.dumps({"id": attempt, "candidate": candidate, "h": h}, indent=2) + "\n"
                    )
                    fit_started = time.perf_counter()
                    pred, model, measurement = fair.fit_predict(
                        df, h, start, CUTOFF, tasks, candidate, settings, event_sink=fair.fit_ledger(out)
                    )
                    validate_frame(pred)
                    if (
                        not pred[KEY].equals(tasks[KEY])
                        or not np.isfinite(pred.yhat).all()
                        or not np.array_equal(pred.target_imputed, tasks.target_imputed)
                        or not np.allclose(pred.y, tasks.y, rtol=0, atol=0, equal_nan=True)
                    ):
                        raise ValueError("Pilot forecast schedule inconsistent")
                    if pd.Timestamp(measurement["latest_fit_target"]) >= CUTOFF:
                        raise ValueError("Supervised label crossed fitting cutoff")
                    model_path = out / f"{attempt}.pkl"
                    artifact_started = time.perf_counter()
                    with model_path.open("wb") as handle:
                        pickle.dump(model, handle)
                    pred.to_parquet(out / f"{attempt}_predictions.parquet", index=False)
                    completed.append(
                        {
                            "model": candidate["model"],
                            "candidate_id": candidate["id"],
                            "h": h,
                            "parameters": json.dumps(candidate["parameters"], sort_keys=True),
                            "validation_origin_start": str(tasks.origin.min()),
                            "validation_origin_end": str(tasks.origin.max()),
                            **measurement,
                            "model_bytes": model_path.stat().st_size,
                            "artifact_seconds": time.perf_counter() - artifact_started,
                            "total_attempt_seconds": time.perf_counter() - fit_started,
                            "finite_predictions": True,
                            "complete_keys": True,
                        }
                    )
                    pd.DataFrame(completed).to_csv(out / "measurements.csv", index=False)
                    record["completed_fits"] = len(completed)
                    check()
                    del model, pred
                    gc.collect()
            if (
                fingerprint(dataset.PROCESSED) != observed
                or fingerprint(dataset.MANIFEST) != record["manifest_sha256"]
            ):
                raise ValueError("Dataset changed during pilot")
        record["status"] = "completed"
        record["readiness"] = "CONDITIONAL_PENDING_RESOURCE_ESTIMATE"
    except PressureBlocked as exc:
        record["status"] = "blocked_memory_pressure"
        record["reason"] = str(exc)
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["reason"] = "Interrupted; completed measurements and active-fit ledger retained"
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["wall_seconds"] = time.perf_counter() - started
        record["peak_rss_bytes"] = fair.peak_rss()
        record["completed_fits"] = len(completed)
        (out / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {"output": str(out), "status": record["status"], "completed_fits": record["completed_fits"]},
            indent=2,
        )
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=fair.DEFAULT)
    parser.add_argument("--output", type=Path, help="new run directory; existing directories are rejected")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute-pilot", action="store_true")
    execution.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(
        json.loads(args.config.read_text()),
        args.execute_pilot,
        args.output,
        preflight_only=args.preflight_only,
    )

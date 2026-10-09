"""Minimal file-based experiment tracking.

Every experiment run writes `results/runs/<experiment>/<utc timestamp>/run.json`
holding what is needed to reproduce it: configuration, data version, splits,
seed, hardware, package versions and the git commit of the code that ran.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import pandas as pd

from gridcast import config

PACKAGES = ["numpy", "pandas", "scikit-learn", "statsmodels", "lightgbm", "torch", "scipy"]


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=config.ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def hardware() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "python": sys.version.split()[0]}
    try:
        import torch

        info["torch_device"] = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_device"] = None
    if sys.platform == "darwin":
        try:
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return info


def versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def data_manifest() -> dict:
    p = config.DATA_PROCESSED / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fold_dict(fold) -> dict:
    return {k: str(getattr(fold, k)) for k in ("name", "train_start", "val_start", "cal_start", "test_start", "test_end")}


class Run:
    def __init__(self, experiment: str, params: dict | None = None, folds: list | None = None, seed: int = 0):
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        self.dir = config.RUNS / experiment / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        man = data_manifest()
        self.record = {
            "experiment": experiment,
            "started_utc": stamp,
            "seed": seed,
            "params": params or {},
            "git": git_state(),
            "hardware": hardware(),
            "versions": versions(),
            "dataset_version": man.get("dataset_version"),
            "eia_raw_sha256": man.get("eia_raw_sha256"),
            "region": config.REGION_NAME,
            "time_index": "UTC, hour ending",
            "local_time_zone": config.LOCAL_TZ,
            "forecast_origin": "every hour in the test block; origin t knows demand through the hour ending at t",
            "folds": [fold_dict(f) for f in (folds or [])],
            "metrics": {},
            "artifacts": [],
        }
        self._t0 = time.perf_counter()

    def log(self, **metrics) -> None:
        self.record["metrics"].update(metrics)

    def artifact(self, path: Path) -> None:
        self.record["artifacts"].append(str(Path(path).resolve().relative_to(config.ROOT)))

    def finish(self) -> Path:
        self.record["wall_seconds"] = round(time.perf_counter() - self._t0, 2)
        out = self.dir / "run.json"
        out.write_text(json.dumps(self.record, indent=2, default=str))
        return out

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.record["status"] = "failed" if exc_type else "completed"
        self.finish()
        return False

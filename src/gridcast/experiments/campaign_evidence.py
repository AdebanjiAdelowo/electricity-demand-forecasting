"""Immutable fit bundles, atomic status files and approved reference verification."""

from __future__ import annotations

import json
import os
import pickle
import uuid
from pathlib import Path

import pyarrow.parquet as pq

from gridcast import config
from gridcast.data import snapshot
from gridcast.experiments.revalidate import fingerprint


class PersistenceError(RuntimeError):
    """Artifact failure must abort, never disqualify a scientific candidate."""


def baseline_dir():
    return Path(
        os.environ.get("GRIDCAST_BASELINE_DIR", config.RESULTS / "forecasting_baselines/20261012T093203Z")
    )


def verify_references(directory=None, approval=None):
    directory = Path(directory) if directory is not None else baseline_dir()
    approval = approval or json.loads((config.ROOT / "configs/approved_fair_reference.json").read_text())
    result = {"directory": str(directory), "files": {}, "problems": []}
    for name, expected in approval["files"].items():
        path = directory / name
        if not path.is_file():
            result["problems"].append(f"missing reference {name}")
            continue
        digest = fingerprint(path)
        result["files"][name] = digest
        if digest != expected["sha256"]:
            result["problems"].append(f"reference SHA-256 mismatch: {name}")
            continue
        if name.endswith(".parquet"):
            meta = pq.ParquetFile(path)
            if (
                meta.metadata.num_rows != expected["rows"]
                or {f.name: str(f.type) for f in meta.schema_arrow} != expected["schema"]
            ):
                result["problems"].append(f"reference schema/rows mismatch: {name}")
    if (
        "run.json" in result["files"]
        and result["files"]["run.json"] == approval["files"]["run.json"]["sha256"]
    ):
        data = json.loads((directory / "run.json").read_text())["data"]
        approved = snapshot.approved()
        if any(data[k] != approved[k] for k in ("processed_sha256", "manifest_sha256")):
            result["problems"].append("reference provenance disagrees with approved snapshot")
    return {**result, "ok": not result["problems"]}


def atomic_write(path, writer, replace=False):
    """One writer per new run. Publish a fully flushed file by same-directory rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial." + uuid.uuid4().hex)
    writer(temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    # Some Drive/FUSE filesystems do not implement directory fsync.
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_json(path, value, replace=False):
    atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2, default=str) + "\n"), replace)


def stage(out, name):
    import time

    record = {"name": name, "started_monotonic": time.monotonic()}
    with (Path(out) / "stages.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    write_json(Path(out) / "stage.json", record, replace=True)


def save_model(path, model):
    def dump(p):
        with p.open("wb") as handle:
            pickle.dump(model, handle)

    atomic_write(path, dump)


def save_fit(out, phase, identity, predictions, model, record):
    """Manifest is the commit marker; a missing manifest means an incomplete fit bundle."""
    directory = Path(out) / "fit_artifacts" / phase / identity
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write(directory / "predictions.parquet", lambda p: predictions.to_parquet(p, index=False))
    if model is not None:
        save_model(directory / "model.pkl", model)
    write_json(directory / "measurement.json", record)
    write_json(
        directory / "complete.json",
        {"status": "complete", "files": {p.name: fingerprint(p) for p in directory.iterdir() if p.is_file()}},
    )


def verify_fit_bundles(out):
    complete, incomplete = [], []
    root = Path(out) / "fit_artifacts"
    for directory in sorted(root.glob("*/*")):
        manifest = directory / "complete.json"
        try:
            record = json.loads(manifest.read_text())
            valid = (
                record.get("status") == "complete"
                and bool(record.get("files"))
                and all(
                    (directory / name).is_file() and fingerprint(directory / name) == digest
                    for name, digest in record["files"].items()
                )
            )
        except (OSError, ValueError):
            valid = False
        (complete if valid else incomplete).append(str(directory.relative_to(out)))
    return {"complete": complete, "incomplete": incomplete}

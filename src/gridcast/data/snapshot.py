"""Fail-closed verification of the approved immutable EIA-930 snapshot.

The approved fingerprints are committed so that a fresh checkout (for example
on Colab) can verify the data without the ignored local benchmark archive.
Schema checks read Parquet metadata only; no demand values are decoded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from gridcast import config
from gridcast.experiments.revalidate import fingerprint

APPROVED = config.ROOT / "configs/approved_dataset_snapshot.json"
BENCHMARK_RUN = config.RESULTS / "forecasting_baselines/20261012T093203Z/run.json"


class SnapshotMismatch(ValueError):
    pass


def approved(path: Path = APPROVED, benchmark: Path = BENCHMARK_RUN) -> dict:
    record = json.loads(path.read_text())
    if benchmark.exists():
        local = json.loads(benchmark.read_text())["data"]
        if (local["processed_sha256"], local["manifest_sha256"]) != (
            record["processed_sha256"],
            record["manifest_sha256"],
        ):
            raise SnapshotMismatch("Committed approval disagrees with the local benchmark record")
    return record


def expected_processed_sha256() -> str:
    return approved()["processed_sha256"]


def schema_problems(processed: Path, record: dict) -> list[str]:
    meta = pq.ParquetFile(processed)
    schema = meta.schema_arrow
    problems = []
    for name, kind in {**record["required_columns"], record["index"]: record["index_type"]}.items():
        if name not in schema.names:
            problems.append(f"missing column {name}")
        elif str(schema.field(name).type) != kind:
            problems.append(f"{name} has type {schema.field(name).type}, expected {kind}")
    if meta.metadata.num_rows != record["rows"]:
        problems.append(f"{meta.metadata.num_rows} rows, expected {record['rows']}")
    if problems:
        return problems
    index = schema.names.index(record["index"])
    demand = schema.names.index("demand")
    stats = [meta.metadata.row_group(i).column(index).statistics for i in range(meta.num_row_groups)]
    if any(s is None or not s.has_min_max for s in stats):
        return ["timestamp statistics unavailable"]
    start, end = min(s.min for s in stats), max(s.max for s in stats)
    if str(start) != record["start"] or str(end) != record["end"]:
        problems.append(f"coverage {start} to {end}, expected {record['start']} to {record['end']}")
    nulls = [meta.metadata.row_group(i).column(demand).statistics for i in range(meta.num_row_groups)]
    if any(s is None or s.null_count != 0 for s in nulls):
        problems.append("demand nulls present or unverifiable")
    return problems


def verify(processed: Path | None = None, manifest: Path | None = None, record: dict | None = None) -> dict:
    """Return a verification record; `ok` is False on any missing file or mismatch."""
    record = record or approved()
    processed = processed or config.ROOT / record["processed_path"]
    manifest = manifest or config.ROOT / record["manifest_path"]
    result = {"dataset_version": record["dataset_version"], "processed": str(processed), "problems": []}
    for path in (processed, manifest):
        if not path.is_file():
            result["problems"].append(f"missing {path}")
    if result["problems"]:
        return {**result, "ok": False}
    result["processed_sha256"] = fingerprint(processed)
    result["manifest_sha256"] = fingerprint(manifest)
    if result["processed_sha256"] != record["processed_sha256"]:
        result["problems"].append("processed dataset SHA-256 differs from approved snapshot")
    if result["manifest_sha256"] != record["manifest_sha256"]:
        result["problems"].append("manifest SHA-256 differs from approved snapshot")
    try:
        result["problems"] += schema_problems(processed, record)
    except Exception as exc:
        result["problems"].append(f"unreadable Parquet metadata: {type(exc).__name__}: {exc}")
    try:
        provenance = json.loads(manifest.read_text())
        for key in ("dataset_version", "rows", "start", "end", "eia_raw_sha256"):
            if provenance.get(key) != record[key]:
                result["problems"].append(f"manifest {key} differs from approved snapshot")
        if provenance.get("time_index") != "UTC, hour ending" or "licenses" not in provenance:
            result["problems"].append("manifest lacks UTC hour-ending index or licence provenance")
        if provenance.get("quality", {}).get("n_imputed") != record["n_imputed"]:
            result["problems"].append("manifest imputation count differs from approved snapshot")
        result["provenance"] = {k: provenance.get(k) for k in ("eia_source", "licenses", "time_index")}
    except Exception as exc:
        result["problems"].append(f"unreadable manifest: {type(exc).__name__}: {exc}")
    return {**result, "ok": not result["problems"]}

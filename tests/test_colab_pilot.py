"""No-fit tests for Linux/Colab admission, dataset gating, supervision and export."""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridcast import config, resources
from gridcast.data import snapshot
from gridcast.experiments import colab_pilot as colab
from gridcast.experiments import fair_model_comparison as fair
from gridcast.experiments import fair_resource_pilot as pilot

GIB = 1024**3
NOTEBOOK = config.ROOT / "notebooks/gridcast_fair_comparison_colab.ipynb"


def write(root: Path, relative: str, text: str):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def linux_tree(root, available=10 * GIB, limit=12 * GIB, usage=1 * GIB, rss=200 * 1024**2, psi=0.0):
    if available is not None:
        write(
            root, "proc/meminfo", f"MemTotal: {13 * GIB // 1024} kB\nMemAvailable: {available // 1024} kB\n"
        )
    if rss is not None:
        write(root, "proc/self/status", f"Name: python\nVmRSS: {rss // 1024} kB\n")
    if limit is not None:
        write(root, "proc/self/cgroup", "0::/\n")
        write(root, "sys/fs/cgroup/memory.max", str(limit))
        write(root, "sys/fs/cgroup/memory.current", str(usage))
    if psi is not None:
        write(root, "proc/pressure/memory", f"some avg10={psi} avg60=0 avg300=0 total=0\n")
    write(root, "sys/fs/cgroup/cpu.max", "200000 100000")
    return root


THREADS = resources.thread_environment(2)


def snap(root, environ=None):
    return resources.linux_snapshot(root, {**THREADS, **(environ or {})}, root)


def verdict(root, environ=None):
    return resources.assess(snap(root, environ), pilot.MEMORY_MIB, 2)


def test_linux_admission_ready_and_conservative_thresholds(tmp_path):
    assert verdict(linux_tree(tmp_path / "ok"))["verdict"] == "READY"
    low = verdict(linux_tree(tmp_path / "low", available=2 * GIB))
    assert low["verdict"] == "BLOCKED" and "required" in low["blocking"][0]
    stalled = verdict(linux_tree(tmp_path / "psi", psi=25.0))
    assert stalled["verdict"] == "BLOCKED" and any("stall" in b for b in stalled["blocking"])


def test_large_host_never_overrides_small_container_limit(tmp_path):
    root = linux_tree(tmp_path, available=50 * GIB, limit=3 * GIB, usage=1 * GIB)
    record = snap(root)
    assert record["effective_available_bytes"] == 2 * GIB
    assert resources.assess(record, pilot.MEMORY_MIB, 2)["verdict"] == "BLOCKED"


def test_cgroup_v2_nested_minimum_v1_and_unlimited(tmp_path):
    v2 = tmp_path / "v2"
    write(v2, "proc/self/cgroup", "0::/outer/inner\n")
    write(v2, "sys/fs/cgroup/outer/memory.max", str(4 * GIB))
    write(v2, "sys/fs/cgroup/outer/inner/memory.max", "max")
    write(v2, "sys/fs/cgroup/outer/inner/memory.current", str(GIB))
    assert resources.cgroup_memory(v2) == {"version": 2, "limit_bytes": 4 * GIB, "usage_bytes": GIB}
    v1 = tmp_path / "v1"
    write(v1, "sys/fs/cgroup/memory/memory.limit_in_bytes", "9223372036854771712")
    write(v1, "sys/fs/cgroup/memory/memory.usage_in_bytes", str(GIB))
    group = resources.cgroup_memory(v1)
    assert group["version"] == 1 and group["limit_bytes"] >= resources.UNLIMITED
    root = tmp_path / "root"
    write(root, "proc/self/cgroup", "0::/\n")
    write(root, "sys/fs/cgroup/cgroup.controllers", "cpu memory")
    assert resources.cgroup_memory(root)["limit_bytes"] == resources.UNLIMITED


def test_missing_memory_sensors_fail_closed(tmp_path):
    assert any("MemAvailable" in b for b in verdict(linux_tree(tmp_path / "a", available=None))["blocking"])
    no_group = verdict(linux_tree(tmp_path / "b", limit=None))
    assert any("cgroup" in b for b in no_group["blocking"]) and no_group["verdict"] == "BLOCKED"
    assert any("RSS" in b for b in verdict(linux_tree(tmp_path / "c", rss=None))["blocking"])
    no_psi = verdict(linux_tree(tmp_path / "d", psi=None))
    assert no_psi["verdict"] == "CONDITIONAL" and not no_psi["blocking"]
    assert resources.assess({"sensor": "darwin"}, 2048, 2)["verdict"] == "BLOCKED"


def test_cpu_only_enforcement(tmp_path):
    gpu = linux_tree(tmp_path / "gpu")
    write(gpu, "dev/nvidia0", "")
    assert any("accelerator" in b for b in verdict(gpu)["blocking"])
    assert verdict(linux_tree(tmp_path / "colab"), {"COLAB_GPU": "1"})["verdict"] == "BLOCKED"
    assert verdict(linux_tree(tmp_path / "tpu"), {"COLAB_TPU_ADDR": "x"})["verdict"] == "BLOCKED"
    assert THREADS["CUDA_VISIBLE_DEVICES"] == ""


def test_thread_limits(tmp_path):
    for bad in (0, 3, 8, "2"):
        with pytest.raises(ValueError):
            resources.thread_environment(bad)
    assert all(THREADS[v] == "2" for v in resources.THREAD_VARIABLES)
    assert resources.thread_problems({**THREADS, "OMP_NUM_THREADS": "4"}, 2) == [
        "OMP_NUM_THREADS=4 exceeds 2"
    ]
    root = linux_tree(tmp_path)
    record = resources.linux_snapshot(root, {}, root)
    assert resources.assess(record, 2048, 2)["verdict"] == "BLOCKED"


def fixture_snapshot(tmp_path, imputed=True, tz="UTC"):
    ix = pd.date_range("2020-01-01", periods=48, freq="h", tz=tz, name="utc_end")
    frame = pd.DataFrame({"demand": np.linspace(40000, 41000, 48), "is_imputed": False}, index=ix)
    if not imputed:
        frame = frame.drop(columns="is_imputed")
    tmp_path.mkdir(parents=True, exist_ok=True)
    processed, manifest = tmp_path / "ercot_hourly.parquet", tmp_path / "manifest.json"
    frame.to_parquet(processed)
    stats = {
        "dataset_version": "fixture",
        "rows": 48,
        "start": str(ix[0]),
        "end": str(ix[-1]),
        "eia_raw_sha256": "raw",
    }
    manifest.write_text(
        json.dumps({**stats, "time_index": "UTC, hour ending", "licenses": {}, "quality": {"n_imputed": 0}})
    )
    record = {
        **stats,
        "processed_sha256": colab.fingerprint(processed),
        "manifest_sha256": colab.fingerprint(manifest),
        "index": "utc_end",
        "index_type": "timestamp[ns, tz=UTC]",
        "required_columns": {"demand": "double", "is_imputed": "bool"},
        "n_imputed": 0,
    }
    return processed, manifest, record


def test_dataset_fixture_passes_and_hash_mismatch_fails(tmp_path):
    processed, manifest, record = fixture_snapshot(tmp_path)
    assert snapshot.verify(processed, manifest, record)["ok"]
    changed = snapshot.verify(processed, manifest, {**record, "processed_sha256": "0" * 64})
    assert not changed["ok"] and any("SHA-256" in p for p in changed["problems"])


def test_dataset_schema_mismatch_fails(tmp_path):
    processed, manifest, record = fixture_snapshot(tmp_path / "a", imputed=False)
    result = snapshot.verify(processed, manifest, {**record})
    assert not result["ok"] and "missing column is_imputed" in result["problems"]
    processed, manifest, record = fixture_snapshot(tmp_path / "b", tz=None)
    result = snapshot.verify(processed, manifest, record)
    assert any("utc_end has type" in p for p in result["problems"])
    processed, manifest, record = fixture_snapshot(tmp_path / "c")
    shifted = snapshot.verify(processed, manifest, {**record, "start": "2019-01-01 00:00:00+00:00"})
    assert any("coverage" in p for p in shifted["problems"]) and not shifted["ok"]


def test_missing_dataset_and_staging_fail_closed(tmp_path, monkeypatch):
    _, _, record = fixture_snapshot(tmp_path / "fixture")
    result = snapshot.verify(tmp_path / "absent.parquet", tmp_path / "absent.json", record)
    assert not result["ok"] and len(result["problems"]) == 2
    with pytest.raises(snapshot.SnapshotMismatch):
        colab.stage_dataset(tmp_path / "fixture")  # fixture differs from the approved snapshot


def test_committed_approval_matches_and_cross_checks_benchmark(tmp_path):
    record = snapshot.approved(benchmark=tmp_path / "absent.json")
    assert record["processed_sha256"] == "1e3acfa6c4070d0dc69d6a2d854964f09afc42b3291221fda646c606249ca5ab"
    bad = tmp_path / "run.json"
    bad.write_text(json.dumps({"data": {"processed_sha256": "0", "manifest_sha256": "0"}}))
    with pytest.raises(snapshot.SnapshotMismatch):
        snapshot.approved(benchmark=bad)


def test_pilot_requires_explicit_authorization(monkeypatch):
    settings = json.loads(fair.DEFAULT.read_text())
    monkeypatch.setattr(colab, "supervise", lambda *a, **k: pytest.fail("Launched without authorization"))
    with pytest.raises(PermissionError, match="authorize"):
        colab.run_pilot(settings, authorized=False)
    monkeypatch.setattr(colab.platform, "system", lambda: "Darwin")
    with pytest.raises(PermissionError, match="macOS"):
        colab.run_pilot(settings, authorized=True)


def test_accidental_full_search_rejected(tmp_path):
    command = colab.pilot_command("python", tmp_path / "x")
    assert command[1:4] == ["-m", colab.PILOT_MODULE, "--execute-pilot"]
    for extra in (["--mode", "full"], ["--authorize-historical-search"], ["--authorize-primary-evaluation"]):
        with pytest.raises(PermissionError):
            colab.assert_pilot_only(command + extra)
    with pytest.raises(PermissionError):
        colab.assert_pilot_only(
            ["python", "-m", "gridcast.experiments.fair_model_comparison", "--execute-pilot"]
        )
    with pytest.raises(SystemExit):
        colab.main(["pilot", "--mode", "full"])
    with pytest.raises(SystemExit):
        colab.main(["full"])


EXPECTED = sorted(colab.EXPECTED_FITS)


def pilot_run_dir(tmp_path, status="completed", fits=EXPECTED, terminal=True):
    out = tmp_path / "resource_pilot_colab_x"
    out.mkdir(parents=True)
    (out / "pilot_plan.json").write_text("{}")
    events = []
    for c, h in fits:
        events += [{"event": "fit_started", "candidate_id": c, "h": h}]
        events += [{"event": "fit_completed", "candidate_id": c, "h": h}]
    (out / "fit_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    rows = [{"candidate_id": c, "h": h, "fit_seconds": 1.0, "model_bytes": 10} for c, h in fits]
    if rows:
        pd.DataFrame(rows).to_csv(out / "measurements.csv", index=False)
    if terminal:
        run = {"status": status, "completed_fits": len(fits), "wall_seconds": 10.0, "peak_rss_bytes": 10}
        (out / "run.json").write_text(json.dumps(run))
    (out / "r01_h1.pkl").write_bytes(b"model")
    (out / "r01_h1_predictions.parquet").write_bytes(b"demand")
    return out


def test_complete_and_interrupted_run_reporting(tmp_path):
    assert colab.inspect(pilot_run_dir(tmp_path / "a"))["label"] == "COMPLETE"
    killed = pilot_run_dir(tmp_path / "b", fits=EXPECTED[:2], terminal=False)
    with (killed / "fit_events.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": "fit_started", "candidate_id": "l11", "h": 1}) + "\n")
    (killed / "supervisor.json").write_text(json.dumps({"exit_code": -9, "killed": "memory: test"}))
    report = colab.inspect(killed)
    assert not report["pilot_complete"] and report["incomplete_fits"] == ["l11_h1"]
    assert any("no terminal run.json" in p for p in report["problems"])
    assert any("killed=memory" in p for p in report["problems"])
    failed = colab.inspect(pilot_run_dir(tmp_path / "c", status="failed"))
    assert failed["label"].startswith("INCOMPLETE")


def test_supervisor_kills_on_memory_and_wall_and_passes_clean_exit():
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    memory = colab.supervise(
        sleeper, None, memory_mib=1, wall_seconds=20, poll=0.05, rss=lambda pid: 2 * 1024**2
    )
    assert memory["killed"].startswith("memory") and memory["exit_code"] != 0
    wall = colab.supervise(sleeper, None, memory_mib=2048, wall_seconds=0.3, poll=0.05, rss=lambda pid: 0)
    assert wall["killed"].startswith("wall") and wall["supervised_wall_seconds"] < 10
    clean = colab.supervise([sys.executable, "-c", "pass"], None, poll=0.05, rss=lambda pid: 0)
    assert clean["exit_code"] == 0 and clean["killed"] is None
    assert colab.HARD_MEMORY_MIB == 2048 and colab.HARD_WALL_SECONDS == 330


def test_tree_rss_sums_descendants(tmp_path):
    write(tmp_path, "proc/10/stat", "10 (python) S 1 0")
    write(tmp_path, "proc/11/stat", "11 (a b) S 10 0")
    write(tmp_path, "proc/12/stat", "12 (other) S 1 0")
    for pid, kb in ((10, 1000), (11, 500), (12, 9999)):
        write(tmp_path, f"proc/{pid}/status", f"VmRSS: {kb} kB\n")
    assert colab.tree_rss(10, tmp_path) == 1500 * 1024


def test_macos_dispatch_gate_unchanged(monkeypatch):
    monkeypatch.setattr(pilot.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda *a, **k: "2\n")
    record = pilot.pressure()
    assert record["dispatch_pressure_level"] == 2 and "sensor" not in record
    with pytest.raises(pilot.PressureBlocked):
        pilot.admit(record)
    pilot.admit({"dispatch_pressure_level": 1, "meaning": "normal"})
    monkeypatch.setattr(pilot.platform, "system", lambda: "Windows")
    with pytest.raises(pilot.PressureBlocked):
        pilot.pressure()


def test_linux_pilot_blocks_before_data_or_fits(tmp_path, monkeypatch):
    settings = json.loads(fair.DEFAULT.read_text())
    root = linux_tree(tmp_path / "root", available=GIB)
    monkeypatch.setattr(pilot.platform, "system", lambda: "Linux")
    record = snap(root)
    monkeypatch.setattr(pilot.resources, "linux_snapshot", lambda **k: record)
    monkeypatch.setattr(pilot, "load_prefix", lambda: pytest.fail("Data read"))
    monkeypatch.setattr(fair, "fit_predict", lambda *a, **k: pytest.fail("Fit"))
    out = pilot.run(settings, True, tmp_path / "blocked")
    record = json.loads((out / "run.json").read_text())
    assert record["status"] == "blocked_memory_pressure" and record["completed_fits"] == 0
    assert record["pressure_observations"][0]["admission"]["verdict"] == "BLOCKED"


def test_preflight_inspects_without_fitting_and_output_paths(tmp_path, monkeypatch):
    settings = json.loads(fair.DEFAULT.read_text())
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(pilot, "load_prefix", lambda: pytest.fail("Labels read during preflight"))
    monkeypatch.setattr(fair, "fit_predict", lambda *a, **k: pytest.fail("Fit during preflight"))
    monkeypatch.setattr(colab.tracking, "git_state", lambda: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(colab.snapshot, "verify", lambda: {"ok": True, "problems": []})
    ready = ({"sensor": "linux"}, {"verdict": "READY", "blocking": [], "warnings": []})
    monkeypatch.setattr(colab, "resource_record", lambda *a, **k: ready)
    out, record = colab.preflight(settings, expected_commit="abc")
    assert record["verdict"] == "READY" and record["fits_executed"] == 0
    assert out.parent == tmp_path / "results" / "fair_model_comparison"
    assert out.name.startswith("colab_preflight_") and ":" not in out.name
    assert json.loads((out / "preflight.json").read_text())["pilot_plan"]["planned_fits"] == 6
    _, wrong = colab.preflight(settings, expected_commit="def")
    assert wrong["verdict"] == "BLOCKED"
    with pytest.raises(FileExistsError):
        colab.preflight(settings, output=out)


def test_export_is_lightweight_and_append_only(tmp_path):
    run = pilot_run_dir(tmp_path)
    destination = tmp_path / "persistent"
    with pytest.raises(FileNotFoundError):
        colab.export(run, destination)
    destination.mkdir()
    archive = colab.export(run, destination)
    names = zipfile.ZipFile(archive).namelist()
    assert f"{run.name}/run.json" in names and f"{run.name}/inspection.json" in names
    assert f"{run.name}/export_manifest.json" in names
    assert not any(n.endswith((".pkl", ".parquet")) for n in names)
    before = archive.read_bytes()
    with pytest.raises(FileExistsError):
        colab.export(run, destination)
    assert archive.read_bytes() == before


def test_reviewed_configuration_and_fixed_plan_unchanged():
    assert colab.fingerprint(fair.DEFAULT) == colab.REVIEWED_CONFIG_SHA256
    plan = pilot.plan(json.loads(fair.DEFAULT.read_text()))
    assert {(c["id"], h) for c in plan["candidates"] for h in plan["horizons"]} == colab.EXPECTED_FITS


def cells():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    return notebook, notebook["cells"]


def test_notebook_structure_order_and_safe_defaults():
    notebook, all_cells = cells()
    code = ["".join(c["source"]) for c in all_cells if c["cell_type"] == "code"]
    steps = [int(s.split("STEP ", 1)[1].split()[0].rstrip(":")) for s in code if "STEP " in s.splitlines()[0]]
    assert steps == list(range(2, 12))
    assert all(
        c.get("outputs", []) == [] and c.get("execution_count") is None
        for c in all_cells
        if c["cell_type"] == "code"
    )
    text = "".join("".join(c["source"]) for c in all_cells)
    assert "AUTHORIZE_SIX_FIT_PILOT = False" in text
    assert "--authorize-historical-search" not in text and "--mode full" not in text.replace("`", "")
    assert "--authorize-six-fit-pilot" in text and "fair_model_comparison --mode" not in text
    assert not any(s in "".join(code) for s in ("ghp_", "github_pat_", "MyDrive/"))
    assert 'REPO_BUNDLE_PATH = ""' in text and 'EXPORT_DIR = ""' in text and 'DATASET_SOURCE_DIR = ""' in text
    assert notebook["metadata"].get("accelerator") in (None, "None")

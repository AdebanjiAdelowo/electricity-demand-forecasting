"""Colab/Linux CPU wrapper for the six-fit resource pilot; never the full search.

Actions are separate and explicit:

    preflight      environment, lockfile, dataset, resources and fixed plan; no data decode or fit
    pilot          the unchanged six fits, only with --authorize-six-fit-pilot, in a supervised child
    inspect        completion/interruption report for one run directory
    export         lightweight provenance/measurement zip to an explicit destination
    stage-dataset  link a user-provided approved snapshot into an incomplete checkout

The supervising parent kills the child process tree at the nominal 2,048 MiB
ceiling or 30 s after the 300 s pilot cap; the pilot's in-process guards
(1,843 MiB early stop, 300 s) are unchanged. Polling cannot catch a spike
shorter than one interval, and the container OOM killer may act first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import tomllib
import zipfile
from importlib import metadata
from pathlib import Path

import pandas as pd

from gridcast import config, resources, tracking
from gridcast.data import dataset, snapshot
from gridcast.experiments import fair_model_comparison as fair
from gridcast.experiments import fair_resource_pilot as pilot
from gridcast.experiments.revalidate import fingerprint

REVIEWED_CONFIG_SHA256 = "5418e81fa00a9930248d2f1a25166eed826fc139bad06f36fc54a4e3fa805d09"
PILOT_MODULE = "gridcast.experiments.fair_resource_pilot"
FORBIDDEN = {
    "--mode",
    "full",
    "development",
    "--authorize-historical-search",
    "--authorize-primary-evaluation",
}
HARD_WALL_SECONDS = pilot.SECONDS + 30
HARD_MEMORY_MIB = pilot.MEMORY_MIB
POLL_SECONDS = 0.25
RESULT_PREFIX = "GRIDCAST_RESULT "
PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "statsmodels"]
EXPECTED_FITS = {(c, h) for c in ("r01", "l02", "l11") for h in pilot.HORIZONS}
EXPORTABLE = [
    "preflight.json",
    "run.json",
    "pilot_plan.json",
    "configuration.json",
    "measurements.csv",
    "fit_events.jsonl",
    "active_attempt.json",
    "supervisor.json",
    "inspection.json",
]


def results_dir() -> Path:
    return config.RESULTS / "fair_model_comparison"


def stamp() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")


def new_run_dir(prefix: str) -> Path:
    out = results_dir() / f"{prefix}_{stamp()}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    return out


def locked_versions(lock: Path) -> dict[str, set[str]]:
    data = tomllib.loads(lock.read_text())
    out: dict[str, set[str]] = {}
    for package in data.get("package", []):
        out.setdefault(package["name"], set()).add(package.get("version"))
    return out


def environment_record(lock: Path) -> tuple[dict, list[str]]:
    locked = locked_versions(lock)
    installed, problems = {}, []
    for name in PACKAGES:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] not in locked.get(name, set()):
            problems.append(f"{name}=={installed[name]} not in uv.lock {sorted(locked.get(name, []))}")
    try:
        torch = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "installed": installed,
        "torch_installed": torch,
        "lock_sha256": fingerprint(lock),
    }, problems


def resource_record(environ=None, root: Path = Path("/")) -> tuple[dict, dict]:
    system = platform.system()
    if system == "Linux":
        record = resources.linux_snapshot(root, environ, results_dir().parent)
        return record, resources.assess(record, pilot.MEMORY_MIB, resources.MAX_THREADS)
    if system == "Darwin":
        try:
            record = pilot.pressure()
            pilot.admit(record)
            return record, {"verdict": "READY", "blocking": [], "warnings": ["macOS dispatch gate"]}
        except Exception as exc:
            return {"sensor": "darwin"}, {"verdict": "BLOCKED", "blocking": [str(exc)], "warnings": []}
    return {"sensor": system}, {
        "verdict": "BLOCKED",
        "blocking": [f"no verified admission gate for {system}"],
        "warnings": [],
    }


def preflight(settings, expected_commit=None, output=None, environ=None, root: Path = Path("/")):
    """Inspect only: no label decoding, model construction or fitting."""
    environ = os.environ if environ is None else environ
    out = output or new_run_dir("colab_preflight")
    blocking, warnings = [], []
    git = tracking.git_state()
    if expected_commit and git.get("commit") != expected_commit:
        blocking.append(f"checked-out commit {git.get('commit')} differs from pinned {expected_commit}")
    if git.get("dirty"):
        blocking.append("working tree differs from the checked-out commit")
    env, problems = environment_record(config.ROOT / "uv.lock")
    blocking += problems
    config_sha = fingerprint(fair.DEFAULT)
    if config_sha != REVIEWED_CONFIG_SHA256:
        blocking.append("scientific configuration differs from the reviewed protocol")
    try:
        data = snapshot.verify()
    except Exception as exc:
        data = {"ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    blocking += [f"dataset: {p}" for p in data["problems"]]
    system, admission = resource_record(environ, root)
    blocking += admission["blocking"]
    warnings += admission["warnings"]
    plan = pilot.plan(settings)
    if {(c["id"], h) for c in plan["candidates"] for h in plan["horizons"]} != EXPECTED_FITS:
        blocking.append("pilot plan is not the fixed six-fit design")
    verdict = "BLOCKED" if blocking else "CONDITIONAL" if warnings else "READY"
    record = {
        "stage": "colab_preflight",
        "utc": str(pd.Timestamp.now(tz="UTC")),
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "meaning": "READY/CONDITIONAL permit an authorized six-fit pilot attempt; neither is authorization",
        "git": git,
        "expected_commit": expected_commit,
        "runtime": resources.runtime(environ),
        "platform": platform.platform(),
        "environment": env,
        "scientific_configuration_sha256": config_sha,
        "dataset": data,
        "resources": system,
        "admission": admission,
        "pilot_plan": plan,
        "hard_limits": {
            "supervisor_memory_mib": HARD_MEMORY_MIB,
            "supervisor_wall_seconds": HARD_WALL_SECONDS,
            "pilot_wall_seconds": pilot.SECONDS,
            "pilot_early_stop_mib": pilot.EARLY_STOP_MIB,
        },
        "fits_executed": 0,
        "full_search_executed": False,
        "primary_evaluation_executed": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "preflight.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return out, record


def pilot_command(python: str, out: Path) -> list[str]:
    command = [python, "-m", PILOT_MODULE, "--execute-pilot", "--output", str(out)]
    assert_pilot_only(command)
    return command


def assert_pilot_only(command: list[str]) -> None:
    if command[1:3] != ["-m", PILOT_MODULE] or FORBIDDEN & set(command) or "--execute-pilot" not in command:
        raise PermissionError("Only the fixed six-fit resource pilot may be launched from Colab")


def tree_rss(pid: int, root: Path = Path("/")) -> int:
    parents: dict[int, list[int]] = {}
    for stat in (root / "proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            parents.setdefault(int(fields[1]), []).append(int(stat.parent.name))
        except (OSError, IndexError, ValueError):
            continue
    total, pending = 0, [pid]
    while pending:
        current = pending.pop()
        total += resources.process_rss(root, str(current)) or 0
        pending += parents.get(current, [])
    return total


def _oom_first():
    # Prefer the pilot over the notebook kernel if the container runs out of memory.
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass


def _signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def supervise(
    command, env, memory_mib=HARD_MEMORY_MIB, wall_seconds=HARD_WALL_SECONDS, poll=POLL_SECONDS, rss=None
):
    rss = rss or tree_rss
    started = time.monotonic()
    child = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
        preexec_fn=_oom_first if platform.system() == "Linux" else None,
    )
    peak, killed, interrupted = 0, None, False
    try:
        while child.poll() is None:
            used = rss(child.pid)
            peak = max(peak, used)
            if used > memory_mib * resources.MIB:
                killed = f"memory: process tree RSS {used / resources.MIB:.0f} MiB > {memory_mib} MiB"
            elif time.monotonic() - started > wall_seconds:
                killed = f"wall: exceeded {wall_seconds} s"
            if killed:
                _signal_group(child.pid, signal.SIGKILL)
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        interrupted = True
        _signal_group(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            killed = "interrupt: child did not stop within 30 s"
            _signal_group(child.pid, signal.SIGKILL)
    code = child.wait()
    return {
        "exit_code": code,
        "killed": killed,
        "interrupted_by_user": interrupted,
        "supervised_wall_seconds": time.monotonic() - started,
        "sampled_peak_tree_rss_bytes": peak,
        "memory_limit_mib": memory_mib,
        "wall_limit_seconds": wall_seconds,
        "poll_seconds": poll,
        "enforcement": "parent polls child process-tree RSS and kills the process group; not a cgroup reservation",
    }


def run_pilot(settings, authorized=False, expected_commit=None, environ=None):
    if not authorized:
        raise PermissionError("The six-fit pilot requires --authorize-six-fit-pilot")
    if platform.system() != "Linux":
        raise PermissionError(
            "Colab supervisor is Linux-only; macOS keeps the existing pilot command and gate"
        )
    environ = dict(os.environ if environ is None else environ)
    environ.update(resources.thread_environment(settings["threads"]))
    check_dir, check = preflight(settings, expected_commit, environ=environ)
    if check["verdict"] == "BLOCKED":
        raise resources.AdmissionBlocked(f"Preflight BLOCKED ({check_dir}): " + "; ".join(check["blocking"]))
    out = new_run_dir("resource_pilot_colab")
    command = pilot_command(sys.executable, out)
    record = {"command": command, "preflight": str(check_dir), "thread_environment": environ_subset(environ)}
    try:
        record.update(supervise(command, environ))
    finally:
        out.mkdir(parents=True, exist_ok=True)
        (out / "preflight.json").write_text((check_dir / "preflight.json").read_text())
        (out / "supervisor.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    report = inspect(out)
    (out / "inspection.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return out, report


def environ_subset(environ):
    return {k: environ.get(k) for k in (*resources.THREAD_VARIABLES, "CUDA_VISIBLE_DEVICES")}


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def inspect(run_dir: Path) -> dict:
    """Report completed, failed and incomplete fits; partial runs are never a completed pilot."""
    run_dir = Path(run_dir)
    run = _json(run_dir / "run.json")
    supervisor = _json(run_dir / "supervisor.json")
    events = []
    if (run_dir / "fit_events.jsonl").exists():
        for line in (run_dir / "fit_events.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                events.append({"event": "unreadable"})
    measurements = (
        pd.read_csv(run_dir / "measurements.csv")
        if (run_dir / "measurements.csv").exists()
        else pd.DataFrame()
    )

    def ids(kind):
        return {(e.get("candidate_id"), e.get("h")) for e in events if e.get("event") == kind}

    started, completed, failed = ids("fit_started"), ids("fit_completed"), ids("fit_failed")
    measured = {(r.candidate_id, int(r.h)) for r in measurements.itertuples()} if len(measurements) else set()
    problems = []
    if run is None:
        problems.append("no terminal run.json: the pilot process was killed or the session disconnected")
    elif run.get("status") != "completed":
        problems.append(f"pilot status {run.get('status')}: {run.get('reason', '')}".strip())
    if supervisor and (supervisor.get("killed") or supervisor.get("exit_code") not in (0, None)):
        problems.append(f"supervisor exit {supervisor.get('exit_code')}; killed={supervisor.get('killed')}")
    if supervisor and supervisor.get("interrupted_by_user"):
        problems.append("interrupted by user")
    if measured != EXPECTED_FITS:
        problems.append(f"{len(measured)} of 6 planned fits measured")
    if run and run.get("wall_seconds", 0) > pilot.SECONDS:
        problems.append("pilot exceeded the 300 s wall-time ceiling")
    if run and (run.get("peak_rss_bytes") or 0) > pilot.MEMORY_MIB * resources.MIB:
        problems.append("pilot exceeded the 2,048 MiB memory ceiling")
    columns = [
        "candidate_id",
        "h",
        "train_n",
        "n_features",
        "feature_preparation_seconds",
        "fit_seconds",
        "predict_seconds",
        "peak_rss_bytes",
        "model_bytes",
        "total_attempt_seconds",
    ]
    return {
        "run_dir": str(run_dir),
        "pilot_complete": not problems,
        "label": "COMPLETE" if not problems else "INCOMPLETE: not a completed pilot",
        "problems": problems,
        "status": run.get("status") if run else None,
        "wall_seconds": run.get("wall_seconds") if run else None,
        "peak_rss_bytes": run.get("peak_rss_bytes") if run else None,
        "planned_fits": 6,
        "measured_fits": sorted(f"{c}_h{h}" for c, h in measured),
        "incomplete_fits": sorted(f"{c}_h{h}" for c, h in started - completed - failed),
        "failed_fits": sorted(f"{c}_h{h}" for c, h in failed),
        "active_attempt": _json(run_dir / "active_attempt.json"),
        "supervisor": supervisor,
        "measurements": measurements[[c for c in columns if c in measurements]].to_dict("records"),
    }


def export(run_dir: Path, destination: Path) -> Path:
    """Zip lightweight provenance only; models and prediction tables (raw demand) are excluded."""
    run_dir, destination = Path(run_dir), Path(destination)
    if not destination.is_dir():
        raise FileNotFoundError(f"Export destination {destination} must be an existing directory")
    archive = destination / f"{run_dir.name}_provenance.zip"
    if archive.exists():
        raise FileExistsError(f"{archive} exists; exports never overwrite")
    report = inspect(run_dir) if (run_dir / "pilot_plan.json").exists() else None
    members = {name: (run_dir / name).read_bytes() for name in EXPORTABLE if (run_dir / name).is_file()}
    if report is not None:
        members["inspection.json"] = (json.dumps(report, indent=2, default=str) + "\n").encode()
    if not members:
        raise FileNotFoundError(f"No exportable provenance in {run_dir}")
    manifest = {
        "run_dir": run_dir.name,
        "exported_utc": str(pd.Timestamp.now(tz="UTC")),
        "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in members.items()},
        "excluded": "fitted models (*.pkl) and prediction tables (*.parquet, contain observed demand)",
    }
    partial = archive.with_suffix(".partial")
    with zipfile.ZipFile(partial, "x", zipfile.ZIP_DEFLATED) as handle:
        for name, content in members.items():
            handle.writestr(f"{run_dir.name}/{name}", content)
        handle.writestr(f"{run_dir.name}/export_manifest.json", json.dumps(manifest, indent=2) + "\n")
    partial.rename(archive)
    return archive


def stage_dataset(source_dir: Path) -> dict:
    """Link a user-provided snapshot only if it matches the approved fingerprints exactly."""
    source_dir = Path(source_dir)
    source = {
        "processed": source_dir / dataset.PROCESSED.name,
        "manifest": source_dir / dataset.MANIFEST.name,
    }
    check = snapshot.verify(source["processed"], source["manifest"])
    if not check["ok"]:
        raise snapshot.SnapshotMismatch("; ".join(check["problems"]))
    for key, target in (("processed", dataset.PROCESSED), ("manifest", dataset.MANIFEST)):
        if target.exists() or target.is_symlink():
            if fingerprint(target) != check[f"{key}_sha256"]:
                raise snapshot.SnapshotMismatch(f"{target} exists with different content; not replaced")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source[key].resolve())
    return snapshot.verify()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=fair.DEFAULT)
    actions = parser.add_subparsers(dest="action")
    check = actions.add_parser("preflight")
    check.add_argument("--expected-commit")
    run = actions.add_parser("pilot")
    run.add_argument("--authorize-six-fit-pilot", action="store_true")
    run.add_argument("--expected-commit")
    look = actions.add_parser("inspect")
    look.add_argument("run_dir", type=Path)
    out = actions.add_parser("export")
    out.add_argument("run_dir", type=Path)
    out.add_argument("--destination", type=Path, required=True)
    stage = actions.add_parser("stage-dataset")
    stage.add_argument("source_dir", type=Path)
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text())
    if args.action is None:
        print(json.dumps(pilot.plan(settings), indent=2, default=str))
        return 0
    if args.action == "preflight":
        path, record = preflight(settings, args.expected_commit)
        result = {"output": str(path), **{k: record[k] for k in ("verdict", "blocking", "warnings")}}
        print(json.dumps(result, indent=2))
        print(RESULT_PREFIX + json.dumps(result))
        return 3 if record["verdict"] == "BLOCKED" else 0
    if args.action == "pilot":
        path, report = run_pilot(settings, args.authorize_six_fit_pilot, args.expected_commit)
        result = {"output": str(path), **{k: report[k] for k in ("label", "problems")}}
        print(json.dumps(result, indent=2))
        print(RESULT_PREFIX + json.dumps(result))
        return 0 if report["pilot_complete"] else 4
    if args.action == "inspect":
        print(json.dumps(inspect(args.run_dir), indent=2, default=str))
        return 0
    if args.action == "export":
        print(export(args.run_dir, args.destination))
        return 0
    print(json.dumps(stage_dataset(args.source_dir), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

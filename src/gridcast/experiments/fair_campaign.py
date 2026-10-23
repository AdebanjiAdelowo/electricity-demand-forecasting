"""Separately authorized Linux campaign supervision. Defaults to no-fit preflight.

All worker outputs and logs live directly in a new operator-selected persistent
run directory. A 0.25 s polling watchdog covers startup, native fits and writes;
it is not an OS reservation and can miss shorter memory spikes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from gridcast import config, resources, tracking
from gridcast.data import snapshot
from gridcast.experiments import campaign_evidence as evidence
from gridcast.experiments import colab_pilot
from gridcast.experiments import fair_model_comparison as fair
from gridcast.experiments.revalidate import fingerprint

MODULE = "gridcast.experiments.fair_model_comparison"
REVIEWED_SHA = colab_pilot.REVIEWED_CONFIG_SHA256


def preflight(settings, mode="full", expected_commit=None):
    fair.validate_settings(settings)
    blocking, warnings = [], []
    git = tracking.git_state()
    if not expected_commit or git.get("commit") != expected_commit or git.get("dirty"):
        blocking.append("clean checkout at an explicitly pinned commit required")
    if fingerprint(fair.DEFAULT) != REVIEWED_SHA or settings != json.loads(fair.DEFAULT.read_text()):
        blocking.append("configuration differs from the frozen experiment")
    environment, errors = colab_pilot.environment_record(config.ROOT / "uv.lock")
    blocking += errors
    if sys.version_info[:2] != (3, 12):
        blocking.append("campaign requires the frozen Python 3.12 project environment")
    if environment.get("torch_installed") is not None:
        blocking.append("use the core project environment without torch/GPU extras")
    try:
        data = snapshot.verify()
    except Exception as exc:
        data = {"ok": False, "problems": [f"dataset verification: {type(exc).__name__}: {exc}"]}
    blocking += data["problems"]
    try:
        reference = evidence.verify_references() if mode == "full" else {"ok": True, "not_required": True}
    except Exception as exc:
        reference = {"ok": False, "problems": [f"reference verification: {type(exc).__name__}: {exc}"]}
    blocking += reference.get("problems", [])
    system, admission = colab_pilot.resource_record()
    blocking += admission["blocking"]
    warnings += admission["warnings"]
    if sys.platform != "linux":
        blocking.append("external campaign execution requires Linux; no macOS gate is bypassed")
    warnings.append("LightGBM 30-minute budget remains at risk; explicit acceptance required before launch")
    return {
        "stage": "full_campaign_preflight",
        "git": git,
        "environment": environment,
        "dataset": data,
        "references": reference,
        "resources": system,
        "blocking": blocking,
        "warnings": warnings,
        "verdict": "BLOCKED" if blocking else "CONDITIONAL",
        "fits_executed": 0,
        "scientific_configuration_sha256": REVIEWED_SHA,
        "plan": fair.plan(settings),
    }


def command(python, output, mode, historical=False, primary=False, accept_budget_risk=False):
    if mode not in ("development", "full"):
        raise ValueError("select development or full explicitly")
    fair.authorize(mode, historical, primary)
    if not accept_budget_risk:
        raise PermissionError("separate explicit resource-budget risk decision required")
    args = [
        str(python),
        "-m",
        MODULE,
        "--mode",
        mode,
        "--output",
        str(output),
        "--authorize-historical-search",
    ]
    if mode == "full":
        args.append("--authorize-primary-evaluation")
    return args


def stage_problem(worker, seconds, now=None):
    """Read every transition, including stages shorter than one supervisor poll."""
    now = time.monotonic() if now is None else now
    path = Path(worker) / "stages.jsonl"
    if not path.exists():
        return None  # startup still covered by the campaign deadline
    try:
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        names = [e["name"] for e in entries]
        allowed = ["startup", "search_ridge", "search_lightgbm", "selection", "primary", "finished"]
        # Development omits primary. No stage can be reset to extend its deadline.
        if (
            not names
            or names[0] != "startup"
            or len(set(names)) != len(names)
            or names != sorted(names, key=allowed.index)
        ):
            return "invalid or reset stage sequence"
        for i, entry in enumerate(entries):
            start = float(entry["started_monotonic"])
            end = float(entries[i + 1]["started_monotonic"]) if i + 1 < len(entries) else now
            if not 0 <= start <= end <= now + 1:
                return "invalid stage clock"
            if entry["name"].startswith("search_") and end - start >= seconds:
                return f"{entry['name']} reached {seconds} s model-search cap"
    except (OSError, ValueError, KeyError, TypeError):
        return "stage journal unreadable; cannot enforce model deadline"
    return None


def supervise(args, env, output, settings, rss=None, poll=0.25):
    output = Path(output)
    rss = rss or colab_pilot.tree_rss
    started, peak, reason = time.monotonic(), 0, None
    child = None
    status = {
        "status": "incomplete",
        "memory_limit_mib": settings["memory_mib"],
        "campaign_seconds": settings["campaign_seconds"],
        "model_search_seconds": settings["model_search_seconds"],
        "poll_seconds": poll,
        "command": args,
    }
    evidence.write_json(output / "supervisor.json", status)
    try:
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            child = subprocess.Popen(args, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            while True:
                code = child.poll()
                now = time.monotonic()
                used = rss(child.pid) if code is None else 0
                peak = max(peak, used)
                if code is None and used <= 0 and child.poll() is None:
                    reason = "process-tree RSS sensor unavailable"
                elif peak >= settings["memory_mib"] * resources.MIB:
                    reason = "process-tree RSS reached memory ceiling"
                elif now - started >= settings["campaign_seconds"]:
                    reason = "campaign deadline reached (includes startup/loading/finalization)"
                else:
                    reason = stage_problem(output / "experiment", settings["model_search_seconds"], now)
                if reason or code is not None:
                    break
                time.sleep(poll)
            if reason:
                colab_pilot._signal_group(child.pid, signal.SIGKILL)
            status.update(exit_code=child.wait(), reason=reason, status="terminated" if reason else "exited")
    except BaseException as exc:
        if child is not None:
            colab_pilot._signal_group(child.pid, signal.SIGKILL)
            child.wait()
        status.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", reason=str(exc)
        )
        raise
    finally:
        status.update(wall_seconds=time.monotonic() - started, sampled_peak_tree_rss_bytes=peak)
        evidence.write_json(output / "supervisor.json", status, replace=True)
    return status


def inspect(output, require_manifest=True):
    """Keep partial evidence visible even if SIGKILL prevented terminal metadata."""
    output = Path(output)
    worker = output / "experiment"
    problems = []

    def read(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            problems.append(f"missing/unreadable {path.name}")
            return {}

    supervisor = read(output / "supervisor.json")
    run = read(worker / "run.json")
    bundles = evidence.verify_fit_bundles(worker)
    search = sum("/search/" in f"/{p}" for p in bundles["complete"])
    primary = sum("/primary/" in f"/{p}" for p in bundles["complete"])
    settings = json.loads(fair.DEFAULT.read_text())
    expected = {
        f"fit_artifacts/search/{name}_{candidate['id']}_h{h}_dev{i + 1}"
        for name, choices in fair.candidates(settings).items()
        for candidate in choices
        for h in settings["horizons"]
        for i in range(len(settings["development_blocks"]))
    }
    required = {
        "configuration.json",
        "candidate_scores.csv",
        "fits.csv",
        "selection.json",
        "development_candidate_predictions.parquet",
        "stages.jsonl",
        "fit_events.jsonl",
    }
    if run.get("stage") == "full":
        expected |= {
            f"fit_artifacts/primary/{name}_{fold.name}_h{h}"
            for name in ("ridge", "lightgbm")
            for fold in config.TEST_FOLDS
            for h in settings["horizons"]
        }
        required |= {
            "evaluation_predictions.parquet",
            "paired_metrics.csv",
            "paired_bootstrap.csv",
            "evaluation_fits.csv",
            "key_accounting.csv",
            "thresholds.json",
            "original_reference_predictions.parquet",
            "original_reference_metrics.csv",
            "original_reference_key_accounting.csv",
        }
    elif run.get("stage") != "development":
        problems.append("unknown or missing campaign stage")
    for name in sorted(required):
        if not (worker / name).is_file():
            problems.append(f"missing final evidence: {name}")
    if run.get("status") != "completed" or supervisor.get("exit_code") != 0 or supervisor.get("reason"):
        problems.append("worker/supervisor did not complete normally")
    if set(bundles["complete"]) != expected or bundles["incomplete"]:
        problems.append("missing, incomplete or corrupt per-fit evidence")
    events = []
    if (worker / "fit_events.jsonl").exists():
        try:
            events = [json.loads(line) for line in (worker / "fit_events.jsonl").read_text().splitlines()]
        except ValueError:
            problems.append("fit ledger truncated/unreadable")

    def key(event):
        return (event.get("candidate_id"), event.get("h"), event.get("cutoff"))

    active = {key(e) for e in events if e.get("event") == "fit_started"} - {
        key(e) for e in events if e.get("event") in ("fit_completed", "fit_failed")
    }
    if active:
        problems.append("interrupted fitting attempts recorded")
    if (
        sum(e.get("event") == "fit_started" for e in events) != len(expected)
        or sum(e.get("event") == "fit_completed" for e in events) != len(expected)
        or any(e.get("event") == "fit_failed" for e in events)
    ):
        problems.append("fit ledger does not confirm every planned fit completed")
    if (
        run.get("wall_seconds", float("inf")) > settings["campaign_seconds"]
        or run.get("peak_rss_bytes", float("inf")) > settings["memory_mib"] * resources.MIB
    ):
        problems.append("worker resource measurements missing or over budget")
    for path in worker.glob("fit_artifacts/primary/*/measurement.json"):
        try:
            m = json.loads(path.read_text())
            model = worker / m["model_file"]
            if not model.is_file() or fingerprint(model) != m["model_sha256"]:
                problems.append(f"missing/corrupt primary model: {model.name}")
        except (OSError, ValueError, KeyError):
            problems.append(f"unreadable primary measurement: {path.parent.name}")
    if require_manifest and not (output / "complete.json").is_file():
        problems.append("no campaign completion marker")
    if (output / "complete.json").exists():
        manifest = read(output / "complete.json")
        hashes = manifest.get("sha256", {})
        if not hashes or any(
            not (output / p).is_file() or fingerprint(output / p) != h for p, h in hashes.items()
        ):
            problems.append("campaign completion manifest differs from saved evidence")
    return {
        "status": "COMPLETE" if not problems else "INCOMPLETE",
        "problems": problems,
        "search_fits_preserved": search,
        "primary_fits_preserved": primary,
        "bundles": bundles,
        "interrupted_attempts": sorted([list(k) for k in active], key=str),
    }


def execute(
    settings,
    persistent_root,
    expected_commit,
    mode="full",
    historical=False,
    primary=False,
    accept_budget_risk=False,
):
    # Authorize before touching data or launching a subprocess.
    command(sys.executable, Path("unused"), mode, historical, primary, accept_budget_risk)
    root = Path(persistent_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("select an existing persistent root")
    if resources.runtime(os.environ) == "colab" and (
        not root.is_relative_to(Path("/content/drive")) or not Path("/content/drive").is_mount()
    ):
        raise ValueError("Colab campaign outputs must be on the mounted Drive, not the ephemeral VM")
    check = preflight(settings, mode, expected_commit)
    output = root / ("fair_campaign_" + colab_pilot.stamp())
    output.mkdir(exist_ok=False)
    evidence.write_json(output / "preflight.json", check)
    if check["verdict"] == "BLOCKED":
        return output, {"status": "BLOCKED", "problems": check["blocking"]}
    env = {
        **os.environ,
        **resources.thread_environment(settings["threads"]),
        "GRIDCAST_ROOT": str(config.ROOT),
        "GRIDCAST_CAMPAIGN_PARENT_PID": str(os.getpid()),
        "PYTHONUNBUFFERED": "1",
    }
    args = command(sys.executable, output / "experiment", mode, historical, primary, accept_budget_risk)
    supervise(args, env, output, settings)
    report = inspect(output, require_manifest=False)
    evidence.write_json(output / "inspection.json", report)
    # Completion manifest is published last; missing marker means incomplete.
    if report["status"] == "COMPLETE":
        evidence.write_json(
            output / "complete.json",
            {
                "status": "complete",
                "sha256": {
                    str(p.relative_to(output)): fingerprint(p)
                    for p in output.rglob("*")
                    if p.is_file() and ".partial." not in p.name
                },
            },
        )
    return output, report


def export(output, destination):
    """Lightweight export; full artifacts already live in the persistent run directory."""
    output, destination = Path(output), Path(destination)
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    target = destination / (output.name + "_provenance.zip")
    members = [
        p
        for p in output.rglob("*")
        if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".log") and ".partial." not in p.name
    ]

    def write(path):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for member in members:
                archive.write(member, str(member.relative_to(output)))
            archive.writestr(
                "export_manifest.json",
                json.dumps(
                    {
                        "sha256": {str(p.relative_to(output)): fingerprint(p) for p in members},
                        "excluded": "model/prediction binaries; retained in private persistent campaign directory",
                    },
                    indent=2,
                ),
            )

    evidence.atomic_write(target, write)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("preflight", "execute", "inspect", "export"), nargs="?", default="preflight"
    )
    parser.add_argument("--mode", choices=("development", "full"), default="full")
    parser.add_argument("--expected-commit")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--persistent-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--authorize-historical-search", action="store_true")
    parser.add_argument("--authorize-primary-evaluation", action="store_true")
    parser.add_argument("--accept-budget-risk", action="store_true")
    args = parser.parse_args()
    if args.baseline_dir:
        os.environ["GRIDCAST_BASELINE_DIR"] = str(args.baseline_dir.resolve())
    settings = json.loads(fair.DEFAULT.read_text())
    if args.action == "preflight":
        result = preflight(settings, args.mode, args.expected_commit)
    elif args.action == "execute":
        if not args.persistent_root:
            parser.error("--persistent-root is required")
        output, report = execute(
            settings,
            args.persistent_root,
            args.expected_commit,
            args.mode,
            args.authorize_historical_search,
            args.authorize_primary_evaluation,
            args.accept_budget_risk,
        )
        result = {"output": str(output), **report}
    else:
        if not args.run_dir:
            parser.error("--run-dir is required")
        if args.action == "export":
            if not args.destination:
                parser.error("--destination is required")
            result = {"archive": str(export(args.run_dir, args.destination))}
        else:
            result = inspect(args.run_dir)
    print(json.dumps(result, indent=2, default=str))
    if result.get("status") in ("BLOCKED", "INCOMPLETE") or result.get("verdict") == "BLOCKED":
        raise SystemExit(3)


if __name__ == "__main__":
    main()

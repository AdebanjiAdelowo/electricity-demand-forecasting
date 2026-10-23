"""No historical fitting: filesystem fixtures, mocked fits and tiny OS workers."""

import json
import os
import sys
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from gridcast.experiments import campaign_evidence as io
from gridcast.experiments import colab_pilot
from gridcast.experiments import fair_campaign as campaign
from gridcast.experiments import fair_model_comparison as fair
from gridcast.experiments.revalidate import fingerprint


@pytest.fixture
def settings():
    return json.loads(fair.DEFAULT.read_text())


def test_authorization_and_budget_decision_before_launch(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(campaign, "preflight", lambda *a: pytest.fail("preflight before authorization"))
    for historical, primary, budget in [(False, False, False), (True, False, True), (True, True, False)]:
        with pytest.raises(PermissionError):
            campaign.execute(
                settings, tmp_path, "pin", historical=historical, primary=primary, accept_budget_risk=budget
            )
    assert "--authorize-primary-evaluation" not in campaign.command(
        "python", tmp_path, "development", True, False, True
    )
    with pytest.raises(PermissionError):
        colab_pilot.assert_pilot_only(campaign.command("python", tmp_path, "full", True, True, True))


def reference_fixture(tmp_path, monkeypatch):
    df = pd.DataFrame({"origin": pd.date_range("2020-01-01", periods=2, tz="UTC"), "yhat": [1.0, 2.0]})
    approved = {"processed_sha256": "p", "manifest_sha256": "m"}
    monkeypatch.setattr(io.snapshot, "approved", lambda: approved)
    (tmp_path / "run.json").write_text(json.dumps({"data": approved}))
    for name in ("ridge", "lightgbm"):
        df.to_parquet(tmp_path / f"{name}.parquet", index=False)
    import pyarrow.parquet as pq

    files = {}
    for name in ("run.json", "ridge.parquet", "lightgbm.parquet"):
        files[name] = {"sha256": fingerprint(tmp_path / name)}
        if name.endswith(".parquet"):
            m = pq.ParquetFile(tmp_path / name)
            files[name].update(rows=2, schema={f.name: str(f.type) for f in m.schema_arrow})
    return {"files": files}


def test_reference_missing_hash_schema_and_provenance(tmp_path, monkeypatch):
    approval = reference_fixture(tmp_path, monkeypatch)
    assert io.verify_references(tmp_path, approval)["ok"]
    approval["files"]["ridge.parquet"]["rows"] = 3
    assert "schema/rows" in str(io.verify_references(tmp_path, approval)["problems"])
    approval["files"]["ridge.parquet"]["sha256"] = "wrong"
    assert "SHA-256" in str(io.verify_references(tmp_path, approval)["problems"])
    (tmp_path / "run.json").unlink()
    assert "missing reference run.json" in io.verify_references(tmp_path, approval)["problems"]


def test_atomic_fit_completion_and_corruption(tmp_path):
    pred = pd.DataFrame({"yhat": [1.0]})
    io.save_fit(tmp_path, "search", "a", pred, SimpleNamespace(alpha=1), {"train_n": 2})
    assert io.verify_fit_bundles(tmp_path) == {"complete": ["fit_artifacts/search/a"], "incomplete": []}
    with pytest.raises(FileExistsError):
        io.save_fit(tmp_path, "search", "a", pred, None, {})
    (tmp_path / "fit_artifacts/search/a/predictions.parquet").write_bytes(b"corrupt")
    assert io.verify_fit_bundles(tmp_path)["incomplete"] == ["fit_artifacts/search/a"]


def test_failed_publish_never_commits_complete_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pd.DataFrame, "to_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        io.save_fit(tmp_path, "search", "a", pd.DataFrame(), None, {})
    assert not (tmp_path / "fit_artifacts/search/a/complete.json").exists()
    assert io.verify_fit_bundles(tmp_path)["incomplete"]


def stages(worker, entries):
    worker.mkdir(parents=True, exist_ok=True)
    (worker / "stages.jsonl").write_text(
        "".join(json.dumps({"name": n, "started_monotonic": t}) + "\n" for n, t in entries)
    )


def test_deadline_cannot_be_reset_or_hidden_by_stage_transition(tmp_path):
    stages(tmp_path, [("startup", 1), ("search_ridge", 2)])
    assert "cap" in campaign.stage_problem(tmp_path, 10, 12)
    stages(tmp_path, [("startup", 1), ("search_ridge", 2), ("search_lightgbm", 13)])
    assert "cap" in campaign.stage_problem(tmp_path, 10, 14)
    stages(tmp_path, [("startup", 1), ("search_ridge", 2), ("search_ridge", 3)])
    assert "reset" in campaign.stage_problem(tmp_path, 10, 4)
    (tmp_path / "stages.jsonl").write_text("{bad")
    assert "unreadable" in campaign.stage_problem(tmp_path, 10, 4)


@pytest.mark.parametrize("fault", ["rss", "sensor", "wall", "stage"])
def test_external_supervisor_stops_worker_and_keeps_logs(tmp_path, settings, fault):
    # Tiny sleeping interpreter simulates native execution that never handles Python timers.
    bounded = {
        **settings,
        "campaign_seconds": 0.05 if fault == "wall" else 3,
        "model_search_seconds": 0.01 if fault == "stage" else 1,
    }
    if fault == "stage":
        stages(tmp_path / "experiment", [("startup", 0), ("search_ridge", 0.001)])

    def rss(pid):
        return 3 * 1024**3 if fault == "rss" else 0 if fault == "sensor" else 1024

    report = campaign.supervise(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        dict(os.environ),
        tmp_path,
        bounded,
        rss=rss,
        poll=0.01,
    )
    assert report["reason"] and report["status"] == "terminated" and report["exit_code"] != 0
    assert (tmp_path / "stdout.log").is_file() and (tmp_path / "stderr.log").is_file()
    assert json.loads((tmp_path / "supervisor.json").read_text())["status"] == "terminated"


def test_no_fit_supervisor_normal_exit_and_append_only(tmp_path, settings):
    report = campaign.supervise(
        [sys.executable, "-c", 'print("fixture complete")'],
        dict(os.environ),
        tmp_path,
        settings,
        rss=lambda pid: 1024,
        poll=0.01,
    )
    assert report["exit_code"] == 0 and report["reason"] is None
    assert "fixture complete" in (tmp_path / "stdout.log").read_text()
    with pytest.raises(FileExistsError):
        campaign.supervise([], {}, tmp_path, settings)


def test_missing_terminal_evidence_is_incomplete_and_export_is_verifiable(tmp_path):
    out = tmp_path / "campaign"
    out.mkdir()
    (out / "preflight.json").write_text('{"fits_executed":0}')
    (out / "private_predictions.parquet").write_bytes(b"not exported")
    assert campaign.inspect(out)["status"] == "INCOMPLETE"
    archive = campaign.export(out, tmp_path)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and not any(n.endswith(".parquet") for n in z.namelist())
        manifest = json.loads(z.read("export_manifest.json"))
        assert manifest["sha256"]["preflight.json"] == fingerprint(out / "preflight.json")
    with pytest.raises(FileExistsError):
        campaign.export(out, tmp_path)


def test_mocked_search_saves_each_block_before_selection(tmp_path, settings, monkeypatch):
    df = fair.synthetic_data().iloc[:1800]
    settings["train_start"] = str(df.index[336])
    settings["development_blocks"] = [[str(df.index[1000]), str(df.index[1300])]]
    grid = {"ridge": fair.candidates(settings)["ridge"][:1]}

    def fake(df, h, start, cutoff, tasks, candidate, settings, **kw):
        return tasks.assign(model="ridge", yhat=tasks.y), SimpleNamespace(), {"train_n": 10}

    monkeypatch.setattr(fair, "fit_predict", fake)
    (tmp_path / "configuration.json").write_text(json.dumps(settings))
    chosen, records = fair.search(df, settings, grid, tmp_path, lambda: None)
    assert len(records) == len(chosen) == 4
    assert len(io.verify_fit_bundles(tmp_path)["complete"]) == 4
    assert (tmp_path / "selection.json").exists()
    # Persistence failures abort rather than selecting from an altered candidate pool.
    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.setattr(io, "save_fit", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(io.PersistenceError):
        fair.search(df, settings, grid, second, lambda: None)
    assert not (second / "selection.json").exists()


def test_historical_input_failure_retained_before_data_or_training(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(fair.snapshot, "verify", lambda: {"ok": True, "problems": []})
    monkeypatch.setattr(
        io, "verify_references", lambda: {"ok": False, "problems": ["missing reference fixture"]}
    )
    monkeypatch.setattr(fair, "load_input", lambda: pytest.fail("data decoded before input admission"))
    monkeypatch.setenv("GRIDCAST_CAMPAIGN_PARENT_PID", str(os.getppid()))
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="Reference input preflight"):
        fair.run(settings, "full", True, True, out)
    record = json.loads((out / "run.json").read_text())
    assert (
        record["status"] == "failed"
        and not record["full_benchmark_executed"]
        and not record["primary_evaluation_started"]
    )
    assert not (out / "fit_events.jsonl").exists()


def test_partial_inspection_retains_committed_fits_and_active_attempts(tmp_path):
    worker = tmp_path / "experiment"
    io.save_fit(worker, "search", "ridge_r00_h1_dev1", pd.DataFrame({"yhat": [1.0]}), None, {})
    (worker / "fit_events.jsonl").write_text(
        json.dumps({"event": "fit_started", "candidate_id": "r01", "h": 168, "cutoff": "earlier"}) + "\n"
    )
    report = campaign.inspect(tmp_path)
    assert report["status"] == "INCOMPLETE" and report["search_fits_preserved"] == 1
    assert report["interrupted_attempts"] == [["r01", 168, "earlier"]]


def test_preflight_blocks_errors_instead_of_launching(settings, monkeypatch):
    monkeypatch.setattr(campaign.tracking, "git_state", lambda: {"commit": "pin", "dirty": False})
    monkeypatch.setattr(campaign.colab_pilot, "environment_record", lambda *a: ({}, []))
    monkeypatch.setattr(
        campaign.snapshot, "verify", lambda: (_ for _ in ()).throw(ValueError("bad dataset manifest"))
    )
    monkeypatch.setattr(io, "verify_references", lambda: {"ok": False, "problems": ["missing references"]})
    monkeypatch.setattr(
        campaign.colab_pilot, "resource_record", lambda: ({}, {"blocking": [], "warnings": []})
    )
    report = campaign.preflight(settings, expected_commit="pin")
    assert report["verdict"] == "BLOCKED" and report["fits_executed"] == 0
    assert "bad dataset manifest" in str(report["blocking"]) and "missing references" in report["blocking"]


def test_blocked_execute_persists_preflight_and_never_starts_worker(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(campaign, "preflight", lambda *a: {"verdict": "BLOCKED", "blocking": ["sensor"]})
    monkeypatch.setattr(campaign, "supervise", lambda *a: pytest.fail("worker launched"))
    output, report = campaign.execute(
        settings, tmp_path, "pin", historical=True, primary=True, accept_budget_risk=True
    )
    assert report["status"] == "BLOCKED" and (output / "preflight.json").is_file()
    assert not (output / "experiment").exists()


def test_linux_direct_historical_entrypoint_requires_supervisor(settings, tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("Linux-only worker admission")
    monkeypatch.delenv("GRIDCAST_CAMPAIGN_PARENT_PID", raising=False)
    monkeypatch.setattr(fair.snapshot, "verify", lambda: pytest.fail("input read without supervisor"))
    with pytest.raises(PermissionError, match="supervisor"):
        fair.run(settings, "full", True, True, tmp_path / "run")


def test_complete_mocked_development_manifest_and_later_corruption(tmp_path, settings):
    # Populate all expected identities without constructing or fitting any model.
    worker = tmp_path / "experiment"
    events = []
    for name, choices in fair.candidates(settings).items():
        for c in choices:
            for h in settings["horizons"]:
                for i in range(3):
                    io.save_fit(
                        worker,
                        "search",
                        f"{name}_{c['id']}_h{h}_dev{i + 1}",
                        pd.DataFrame({"yhat": [1.0]}),
                        SimpleNamespace(),
                        {},
                    )
                    for event in ("fit_started", "fit_completed"):
                        events.append({"event": event, "candidate_id": c["id"], "h": h, "cutoff": str(i)})
    (worker / "fit_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for name in ("configuration.json", "selection.json"):
        (worker / name).write_text("{}")
    for name in ("candidate_scores.csv", "fits.csv"):
        (worker / name).write_text("fixture\n")
    pd.DataFrame({"yhat": [1.0]}).to_parquet(
        worker / "development_candidate_predictions.parquet", index=False
    )
    stages(
        worker,
        [("startup", 1), ("search_ridge", 2), ("search_lightgbm", 3), ("selection", 4), ("finished", 5)],
    )
    io.write_json(
        worker / "run.json",
        {"status": "completed", "stage": "development", "wall_seconds": 4, "peak_rss_bytes": 1000},
    )
    io.write_json(tmp_path / "supervisor.json", {"exit_code": 0, "reason": None})
    assert campaign.inspect(tmp_path)["status"] == "INCOMPLETE"  # marker still missing
    assert campaign.inspect(tmp_path, require_manifest=False)["status"] == "COMPLETE"
    io.write_json(
        tmp_path / "complete.json",
        {
            "sha256": {
                str(p.relative_to(tmp_path)): fingerprint(p) for p in tmp_path.rglob("*") if p.is_file()
            }
        },
    )
    assert campaign.inspect(tmp_path)["status"] == "COMPLETE"
    (worker / "selection.json").write_text('{"changed":true}')
    assert "completion manifest differs" in str(campaign.inspect(tmp_path)["problems"])


def test_mocked_primary_fits_publish_models_predictions_and_measurements(tmp_path, settings, monkeypatch):
    df = fair.synthetic_data().iloc[:1800]
    settings["horizons"] = [1]
    grid = fair.candidates(settings)
    chosen = [{"model": name, "h": 1, "candidate_id": choices[0]["id"]} for name, choices in grid.items()]
    (tmp_path / "selection.json").write_text("{}")

    def fake(df, h, start, cutoff, tasks, candidate, settings, **kw):
        return tasks.assign(model=candidate["model"], yhat=tasks.y + 1), SimpleNamespace(), {"train_n": 10}

    monkeypatch.setattr(fair, "fit_predict", fake)
    fair.evaluate(
        df,
        settings,
        grid,
        chosen,
        [("fixture", df.index[336], df.index[1000], df.index[1000], df.index[1300])],
        tmp_path,
        lambda: None,
        synthetic=True,
    )
    assert len(io.verify_fit_bundles(tmp_path)["complete"]) == 2
    for name in grid:
        assert (tmp_path / f"{name}_fixture_h1.pkl").is_file()
        record = json.loads(
            (tmp_path / f"fit_artifacts/primary/{name}_fixture_h1/measurement.json").read_text()
        )
        assert record["model_sha256"] == fingerprint(tmp_path / record["model_file"])
    assert len(pd.read_csv(tmp_path / "evaluation_fits.csv")) == 2


def test_parent_interrupt_kills_worker_and_preserves_status(tmp_path, settings):
    def interrupt(pid):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        campaign.supervise(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            dict(os.environ),
            tmp_path,
            settings,
            rss=interrupt,
        )
    assert json.loads((tmp_path / "supervisor.json").read_text())["status"] == "interrupted"


def test_worker_interrupt_with_truncated_ledger_keeps_terminal_status(tmp_path, settings, monkeypatch):
    def interrupt(df, settings, grid, out, check):
        (out / "fit_events.jsonl").write_text("{interrupted append")
        raise KeyboardInterrupt

    monkeypatch.setattr(fair, "search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fair.run(settings, "smoke", output=tmp_path / "worker")
    record = json.loads((tmp_path / "worker/run.json").read_text())
    assert record["status"] == "interrupted" and record["fit_ledger_incomplete"]
    assert not record["full_benchmark_executed"]

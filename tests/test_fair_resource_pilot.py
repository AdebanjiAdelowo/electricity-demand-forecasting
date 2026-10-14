import json
from types import SimpleNamespace

import pandas as pd
import pytest

from gridcast.experiments import fair_model_comparison as fair
from gridcast.experiments import fair_resource_pilot as pilot


@pytest.fixture
def settings():
    return json.loads(fair.DEFAULT.read_text())


@pytest.fixture
def prefix():
    ix = pd.date_range("2023-06-01", "2023-07-15", inclusive="left", freq="h", tz="UTC")
    return pd.DataFrame({"demand": 40000.0, "is_imputed": False}, index=ix)


def block_other_stages(monkeypatch):
    for function in ["search", "select", "evaluate", "load_input"]:
        monkeypatch.setattr(fair, function, lambda *a, **k: pytest.fail("Other stage accessed"))


def test_plan_is_fixed_predefined_and_does_not_fit(settings, monkeypatch):
    monkeypatch.setattr(fair, "fit_predict", lambda *a, **k: pytest.fail("Unexpected fit"))
    assert pilot.run(settings) is None
    selected = pilot.pilot_candidates(settings)
    assert [p["id"] for p in selected] == ["r01", "l02", "l11"]
    assert selected[1]["parameters"]["n_estimators"] == 600
    assert selected[1]["parameters"]["num_leaves"] == 63
    assert selected[1]["parameters"]["objective"] == "l2"
    assert selected[2]["parameters"]["n_estimators"] == 300
    assert json.loads(fair.DEFAULT.read_text()) == settings


@pytest.mark.parametrize("level", [2, 4, 0, 99])
def test_pressure_rejects_unknown_or_elevated_levels_before_data_or_fits(
    level, settings, monkeypatch, tmp_path
):
    block_other_stages(monkeypatch)
    monkeypatch.setattr(pilot, "pressure", lambda: {"dispatch_pressure_level": level, "meaning": "test"})
    monkeypatch.setattr(pilot, "load_prefix", lambda: pytest.fail("Data read under pressure"))
    monkeypatch.setattr(fair, "fit_predict", lambda *a, **k: pytest.fail("Fit under pressure"))
    out = pilot.run(settings, True, tmp_path / "blocked")
    record = json.loads((out / "run.json").read_text())
    assert record["status"] == "blocked_memory_pressure" and record["completed_fits"] == 0
    assert not record["selection_executed"] and not record["primary_evaluation_executed"]


def test_prefix_loader_passes_pre_test_predicate_and_serial_columns(prefix, monkeypatch):
    calls = []

    def read(*a, **kwargs):
        calls.append(kwargs)
        return prefix

    monkeypatch.setattr(pilot.pd, "read_parquet", read)
    assert pilot.load_prefix() is prefix
    assert calls[0]["columns"] == ["demand", "is_imputed"] and not calls[0]["use_threads"]
    assert calls[0]["filters"][0][0:2] == ("utc_end", "<")
    assert pd.Timestamp(calls[0]["filters"][0][2]) == pilot.END
    bad = prefix.copy()
    bad.index = bad.index + pd.Timedelta(days=365)
    monkeypatch.setattr(pilot.pd, "read_parquet", lambda *a, **k: bad)
    with pytest.raises(ValueError, match="outside"):
        pilot.load_prefix()


def fake_fit(df, h, start, cutoff, tasks, candidate, settings, **kwargs):
    assert cutoff == pilot.CUTOFF and (tasks.target_time < pilot.END).all()
    assert (df.index < pilot.END).all()
    return (
        tasks.assign(model=candidate["model"], yhat=40000.0),
        SimpleNamespace(),
        {
            "fit_seconds": 0.1,
            "predict_seconds": 0.01,
            "latest_fit_target": str(cutoff - pd.Timedelta(hours=1)),
            "train_n": 1000,
            "feature_names": ["fixture"],
            "n_features": 1,
            "feature_preparation_seconds": 0.02,
            "iterations": candidate["parameters"].get("n_estimators", 1),
            "peak_rss_bytes": 1000,
        },
    )


def test_pilot_six_fits_no_search_selection_or_primary_calls(settings, prefix, monkeypatch, tmp_path):
    block_other_stages(monkeypatch)
    monkeypatch.setattr(pilot, "pressure", lambda: {"dispatch_pressure_level": 1, "meaning": "normal"})
    monkeypatch.setattr(pilot, "load_prefix", lambda: prefix)
    monkeypatch.setattr(fair, "fit_predict", fake_fit)
    out = pilot.run(settings, True, tmp_path / "run")
    record = json.loads((out / "run.json").read_text())
    assert record["status"] == "completed" and record["completed_fits"] == 6
    assert not (out / "selection.json").exists()
    assert not record["primary_evaluation_executed"] and not record["full_search_executed"]
    assert set(pd.read_csv(out / "measurements.csv").h) == {1, 168}
    before = (out / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        pilot.run(settings, True, out)
    assert (out / "run.json").read_bytes() == before


def test_interrupted_second_fit_keeps_first_measurement_and_failure_metadata(
    settings, prefix, monkeypatch, tmp_path
):
    block_other_stages(monkeypatch)
    monkeypatch.setattr(pilot, "pressure", lambda: {"dispatch_pressure_level": 1, "meaning": "normal"})
    monkeypatch.setattr(pilot, "load_prefix", lambda: prefix)
    count = [0]

    def interrupted(*args, **kwargs):
        count[0] += 1
        if count[0] == 2:
            raise fair.ResourceLimit("fixture limit")
        return fake_fit(*args, **kwargs)

    monkeypatch.setattr(fair, "fit_predict", interrupted)
    out = tmp_path / "partial"
    with pytest.raises(fair.ResourceLimit):
        pilot.run(settings, True, out)
    record = json.loads((out / "run.json").read_text())
    assert record["status"] == "failed" and record["completed_fits"] == 1
    assert len(pd.read_csv(out / "measurements.csv")) == 1
    assert (out / "r01_h1.pkl").exists() and (out / "active_attempt.json").exists()


def test_cutoff_violation_never_becomes_completed_pilot(settings, prefix, monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "pressure", lambda: {"dispatch_pressure_level": 1, "meaning": "normal"})
    monkeypatch.setattr(pilot, "load_prefix", lambda: prefix)

    def bad(*args, **kwargs):
        pred, model, record = fake_fit(*args, **kwargs)
        record["latest_fit_target"] = str(pilot.CUTOFF)
        return pred, model, record

    monkeypatch.setattr(fair, "fit_predict", bad)
    with pytest.raises(ValueError, match="crossed"):
        pilot.run(settings, True, tmp_path / "cutoff")


def test_pilot_uses_original_caps_with_early_memory_stop(settings):
    assert pilot.SECONDS == 300 and pilot.MEMORY_MIB == 2048
    assert pilot.EARLY_STOP_MIB < pilot.MEMORY_MIB
    assert settings["model_search_seconds"] == 1800 and settings["campaign_seconds"] == 3600
    with pytest.raises(PermissionError):
        fair.authorize("full", True, False)
    with pytest.raises(PermissionError):
        fair.authorize("full", False, True)


def test_preflight_even_when_normal_cannot_read_labels_or_fit(settings, monkeypatch, tmp_path):
    block_other_stages(monkeypatch)
    monkeypatch.setattr(pilot, "pressure", lambda: {"dispatch_pressure_level": 1, "meaning": "normal"})
    monkeypatch.setattr(pilot, "load_prefix", lambda: pytest.fail("Labels read during preflight"))
    monkeypatch.setattr(fair, "fit_predict", lambda *a, **k: pytest.fail("Fit during preflight"))
    out = pilot.run(settings, False, tmp_path / "preflight", preflight_only=True)
    assert json.loads((out / "run.json").read_text())["status"] == "preflight_normal_no_fits"
    with pytest.raises(ValueError, match="either"):
        pilot.run(settings, True, preflight_only=True)


def test_keyboard_interruption_records_terminal_status(settings, prefix, monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "pressure", lambda: {"dispatch_pressure_level": 1, "meaning": "normal"})
    monkeypatch.setattr(pilot, "load_prefix", lambda: prefix)

    def stop(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(fair, "fit_predict", stop)
    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        pilot.run(settings, True, out)
    record = json.loads((out / "run.json").read_text())
    assert record["status"] == "interrupted" and record["completed_fits"] == 0
    assert (out / "active_attempt.json").exists()

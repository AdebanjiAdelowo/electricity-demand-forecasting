import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gridcast.evaluation import fair_comparison as ev
from gridcast.experiments import fair_model_comparison as experiment


@pytest.fixture
def settings():
    return json.loads(experiment.DEFAULT.read_text())


@pytest.fixture
def demand():
    return experiment.synthetic_data().iloc[:1800].copy()


def frame(n=1000):
    origin = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    a = pd.DataFrame(
        {
            "model": "ridge",
            "regime": "none",
            "fold": "f",
            "h": 24,
            "origin": origin,
            "target_time": origin + pd.Timedelta(hours=24),
            "y": 100.0,
            "yhat": 101.0,
            "target_imputed": False,
        }
    )
    return pd.concat([a, a.assign(model="lightgbm", yhat=102.0)], ignore_index=True)


def test_grid_enumeration_and_reference_configuration(settings):
    first = experiment.candidates(settings)
    assert first == experiment.candidates(settings)
    assert len(first["ridge"]) == 5 and len(first["lightgbm"]) == 12
    assert [c["parameters"]["alpha"] for c in first["ridge"]] == [0.1, 1.0, 10.0, 100.0, 1000.0]
    p = first["lightgbm"][2]["parameters"]
    assert p["objective"] == "l2" and p["num_leaves"] == 63 and p["n_estimators"] == 600
    assert experiment.plan(settings)["search_fits"] == {"ridge": 60, "lightgbm": 144}


def test_chronological_blocks_and_primary_access_rejected(settings):
    blocks = experiment.validate_settings(settings)
    assert all(a < b for a, b in blocks) and blocks[-1][1] == pd.Timestamp("2024-04-01", tz="UTC")
    bad = copy.deepcopy(settings)
    bad["development_blocks"][1][0] = "2023-09-01T00:00:00Z"
    with pytest.raises(ValueError, match="Overlapping"):
        experiment.validate_settings(bad)
    bad = copy.deepcopy(settings)
    bad["development_blocks"][-1][1] = "2024-07-01T00:00:00Z"
    with pytest.raises(ValueError, match="primary"):
        experiment.validate_settings(bad)


@pytest.mark.parametrize("h", [1, 24, 48, 168])
def test_fit_targets_precede_cutoff_and_imputed_exclusion(demand, h):
    x, y, m, imputed = experiment.prepare(demand, h)
    cutoff = demand.index[1000]
    imputed.iloc[800] = True
    mask = experiment.fitting_mask(x, y, m, imputed, h, demand.index[336], cutoff)
    assert (x.index[mask] + pd.Timedelta(hours=h) < cutoff).all()
    assert not mask.iloc[800]
    assert not mask.iloc[1000 - h]


@pytest.mark.parametrize("h", [1, 24, 48, 168])
def test_future_demand_does_not_change_past_features(demand, h):
    a = experiment.prepare(demand, h)[0]
    changed = demand.copy()
    changed.loc[changed.index > demand.index[1000], "demand"] += 1e6
    b = experiment.prepare(changed, h)[0]
    pd.testing.assert_frame_equal(a.iloc[:1001], b.iloc[:1001])
    assert not any(c.startswith("w_") for c in a)
    assert {f"oh_month{i}" for i in range(1, 13)} <= set(a)


@pytest.mark.parametrize("model", ["ridge", "lightgbm"])
def test_shared_preprocessing_fits_training_only_and_raw_info_is_identical(demand, settings, model):
    grid = experiment.candidates(settings)
    candidate = copy.deepcopy(grid[model][0])
    if model == "lightgbm":
        candidate["parameters"]["n_estimators"] = 3
    cutoff = demand.index[1000]
    tasks = experiment.schedule(demand, 24, "dev", cutoff, demand.index[1200])
    p, fitted, record = experiment.fit_predict(
        demand, 24, demand.index[336], cutoff, tasks, candidate, settings
    )
    x, y, m, imputed = experiment.prepare(demand, 24)
    tr = experiment.fitting_mask(x, y, m, imputed, 24, demand.index[336], cutoff)
    np.testing.assert_allclose(fitted.named_steps["simpleimputer"].statistics_, x.loc[tr].median().to_numpy())
    transformed = fitted.named_steps["simpleimputer"].transform(x.loc[tr])
    np.testing.assert_allclose(fitted.named_steps["standardscaler"].mean_, transformed.mean(axis=0))
    assert record["preprocessing_fit_n"] == int(tr.sum())
    assert list(fitted.feature_names_in_) == list(x.columns)
    changed = demand.copy()
    changed.loc[changed.index >= cutoff, "demand"] += 1e6
    _, second, _ = experiment.fit_predict(changed, 24, demand.index[336], cutoff, tasks, candidate, settings)
    np.testing.assert_array_equal(
        fitted.named_steps["simpleimputer"].statistics_, second.named_steps["simpleimputer"].statistics_
    )
    np.testing.assert_array_equal(
        fitted.named_steps["standardscaler"].mean_, second.named_steps["standardscaler"].mean_
    )
    assert len(p) == len(tasks) and np.isfinite(p.yhat).all()


def test_selection_primary_mae_ties_and_failed_candidates():
    table = pd.DataFrame(
        dict(
            model=["ridge"] * 3,
            h=[24] * 3,
            candidate_id=["r01", "r00", "r02"],
            complete=[True, True, False],
            mae_mw=[2.0, 2.0, 0.0],
            n=[100] * 3,
        )
    )
    assert experiment.select(table)[0]["candidate_id"] == "r00"
    table["complete"] = False
    with pytest.raises(ValueError, match="No complete"):
        experiment.select(table)


def test_metrics_known_values():
    p = ev.summary(np.array([1.0, 3.0]), np.array([3.0, 2.0]))
    assert p["mae_mw"] == 1.5 and p["rmse_mw"] == pytest.approx(np.sqrt(2.5))
    assert p["signed_mean_error_mw"] == 0.5 and p["median_absolute_error_mw"] == 1.5
    with pytest.raises(ValueError):
        ev.summary([1.0], [np.nan])


def test_matching_keys_targets_and_missing_reports():
    p = frame(100)
    p.loc[2, "target_imputed"] = True
    p.loc[102, "target_imputed"] = True
    p = p.drop(index=104)
    p.loc[105, "yhat"] = np.nan
    canonical = frame(100).iloc[:100].copy()
    canonical.loc[2, "target_imputed"] = True
    paired, accounting = ev.match(p, schedule=canonical)
    assert len(paired) == 97
    assert accounting.missing_key.sum() == 1
    assert accounting.imputed_target.sum() == 2
    assert accounting.nonfinite.sum() == 1
    assert paired.ae_difference.eq(-1).all()
    bad = frame(100)
    bad.loc[100, "y"] += 1
    with pytest.raises(ValueError, match="Different observations"):
        ev.match(bad)
    bad = frame(100)
    bad.loc[100, "target_imputed"] = True
    with pytest.raises(ValueError, match="imputation"):
        ev.match(bad)


def test_duplicate_and_misaligned_key_rejection():
    p = frame(100)
    with pytest.raises(ValueError, match="Duplicate"):
        ev.match(pd.concat([p, p.iloc[:1]], ignore_index=True))
    p.loc[0, "target_time"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="alignment"):
        ev.match(p)


def test_bootstrap_respects_fold_boundaries_and_is_reproducible():
    a = frame(1200)
    b = frame(1200).assign(fold="other")
    b.loc[b.model == "ridge", "yhat"] = 104.0
    paired, _ = ev.match(pd.concat([a, b], ignore_index=True))
    first = ev.bootstrap(paired, replicates=20)
    pd.testing.assert_frame_equal(first, ev.bootstrap(paired, replicates=20))
    # Every stratum has a constant loss difference: fold-preserving sampling
    # cannot change its count-weighted overall mean of (-1 + 2)/2 = .5.
    assert first.ridge_minus_lightgbm_mae_mw.eq(0.5).all()
    np.testing.assert_allclose(first.ci95_low_mw, 0.5)
    np.testing.assert_allclose(first.ci95_high_mw, 0.5)
    assert first.fold_count.eq(2).all()


def test_bootstrap_uses_elapsed_days_including_gaps_and_flags_short_folds():
    p = frame(1200)
    keep = ~p.origin.between(pd.Timestamp("2023-01-10", tz="UTC"), pd.Timestamp("2023-01-15", tz="UTC"))
    paired, _ = ev.match(p[keep])
    result = ev.bootstrap(paired, days=(7,), replicates=10)
    assert result.effective_blocks_approx.iloc[0] == pytest.approx(50 / 7)
    paired, _ = ev.match(frame(48))
    assert ev.bootstrap(paired, days=(7,), replicates=10).status.iloc[0] == "insufficient_fold_span"


def test_extreme_threshold_only_earlier_nonimputed_demand(demand):
    cutoff = demand.index[1000]
    start = demand.index[336]
    threshold = ev.extreme_threshold(demand, start, cutoff)
    changed = demand.copy()
    changed.loc[changed.index >= cutoff, "demand"] = 1e9
    assert ev.extreme_threshold(changed, start, cutoff) == threshold
    changed.loc[changed.index[500], "is_imputed"] = True
    changed.loc[changed.index[500], "demand"] = 1e10
    expected = changed[
        (changed.index >= start) & (changed.index < cutoff) & ~changed.is_imputed
    ].demand.quantile(0.95)
    assert ev.extreme_threshold(changed, start, cutoff) == expected


@pytest.mark.parametrize(
    "mode,historical,primary",
    [("full", False, False), ("full", True, False), ("full", False, True), ("development", False, False)],
)
def test_accidental_historical_execution_rejected_before_data_access(
    settings, monkeypatch, mode, historical, primary
):
    monkeypatch.setattr(
        experiment, "load_input", lambda: pytest.fail("Real data accessed before authorization")
    )
    with pytest.raises(PermissionError):
        experiment.run(settings, mode, historical, primary)


def test_import_and_dry_run_do_not_fit_or_read_historical_observations(settings, monkeypatch):
    monkeypatch.setattr(experiment, "load_input", lambda: pytest.fail("Real data read"))
    monkeypatch.setattr(experiment, "fit_predict", lambda *a, **k: pytest.fail("Training invoked"))
    assert experiment.run(settings) is None


def test_resource_budget_failure_is_safe_and_timer_restored(monkeypatch):
    monkeypatch.setattr(experiment, "peak_rss", lambda: 3 * 1024**3)
    with pytest.raises(experiment.ResourceLimit):
        with experiment.budget(10, 2048):
            pytest.fail("Execution under excessive memory")
    monkeypatch.setattr(experiment, "peak_rss", lambda: 0)
    ticks = iter([0.0, 0.0, 20.0])
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(experiment.ResourceLimit):
        with experiment.budget(10, 2048):
            pass


def test_failed_input_integrity_is_retained_without_overwriting_archives(settings, monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    processed.write_text("original")
    monkeypatch.setattr(experiment.dataset, "PROCESSED", processed)
    monkeypatch.setattr(experiment.dataset, "MANIFEST", tmp_path / "absent")
    # Place temporary test data beneath ROOT for the provenance relative paths.
    monkeypatch.setattr(experiment.config, "ROOT", tmp_path)
    (tmp_path / "uv.lock").write_text("test lock")
    monkeypatch.setattr(experiment, "search", lambda *args: ([], []))

    def mutate(*args, **kwargs):
        processed.write_text("changed fixture")

    monkeypatch.setattr(experiment, "evaluate", mutate)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="Input mutated"):
        experiment.run(settings, "smoke", output=out)
    record = json.loads((out / "run.json").read_text())
    assert record["status"] == "failed" and "Input mutated" in record["error"]
    before = (out / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        experiment.run(settings, "smoke", output=out)
    assert (out / "run.json").read_bytes() == before


def test_candidate_missing_predictions_remain_disqualified(settings, demand, monkeypatch, tmp_path):
    settings["train_start"] = str(demand.index[336])
    settings["development_blocks"] = [[str(demand.index[1000]), str(demand.index[1300])]]
    g = experiment.candidates(settings)
    g = {"ridge": g["ridge"][:1]}

    def missing(df, h, start, cutoff, tasks, candidate, settings, **kwargs):
        return tasks.iloc[:-1].assign(model="ridge", yhat=1.0), SimpleNamespace(), {}

    monkeypatch.setattr(experiment, "fit_predict", missing)
    with pytest.raises(ValueError, match="No complete"):
        experiment.search(demand, settings, g, tmp_path, lambda: None)
    table = pd.read_csv(tmp_path / "candidate_scores.csv")
    assert not table.complete.any() and table.failure.str.contains("keys differ").all()


def test_configuration_cannot_override_cpu_threads_or_expand_budget(settings):
    bad = copy.deepcopy(settings)
    bad["lightgbm_profiles"][0]["device_type"] = "gpu"
    with pytest.raises(ValueError, match="CPU/thread"):
        experiment.candidates(bad)
    bad = copy.deepcopy(settings)
    bad["threads"] = 3
    with pytest.raises(ValueError, match="threads"):
        experiment.validate_settings(bad)
    bad = copy.deepcopy(settings)
    bad["memory_mib"] = 4096
    with pytest.raises(ValueError, match="memory"):
        experiment.validate_settings(bad)


def test_schedule_detects_observation_change_even_when_other_model_is_missing():
    p = frame(100)
    schedule = p.iloc[:100].copy()
    p = p.drop(index=100)
    p.loc[0, "y"] = 200.0
    with pytest.raises(ValueError, match="canonical"):
        ev.match(p, schedule=schedule)


def test_last_candidate_cannot_evade_common_search_cap(settings, demand, monkeypatch, tmp_path):
    settings["train_start"] = str(demand.index[336])
    settings["development_blocks"] = [[str(demand.index[1000]), str(demand.index[1300])]]
    settings["model_search_seconds"] = 0.1
    grid = {"ridge": experiment.candidates(settings)["ridge"][:1]}
    clock = [0.0]
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: clock[0])

    def completed(df, h, start, cutoff, tasks, candidate, settings, **kwargs):
        clock[0] = 1.0
        return tasks.assign(model="ridge", yhat=tasks.y), SimpleNamespace(), {}

    monkeypatch.setattr(experiment, "fit_predict", completed)
    with pytest.raises(experiment.ResourceLimit, match="common model budget"):
        experiment.search(demand, settings, grid, tmp_path, lambda: None)
    assert (tmp_path / "fits.csv").exists() and not (tmp_path / "selection.json").exists()


def test_loader_uses_no_weather_columns_or_parallel_decode(monkeypatch, demand):
    calls = []

    def read(path, **kwargs):
        calls.append(kwargs)
        return demand

    monkeypatch.setattr(experiment.pd, "read_parquet", read)
    assert experiment.load_input() is demand
    assert calls[0] == {"columns": ["demand", "is_imputed"], "use_threads": False}
    demand.index = demand.index.as_unit("us")
    assert len(experiment.prepare(demand, 24)[0]) == len(demand)

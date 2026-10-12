from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest

from gridcast.experiments.export_predictions import (
    evaluation_rows,
    plot_week,
    reproduce_metrics,
    select_weeks,
)


def source():
    idx = pd.date_range("2024-01-01", periods=1000, freq="h", tz="UTC")
    y = 100 + np.sin(np.arange(len(idx)) / 24)
    obs = pd.DataFrame({"demand": y, "demand_raw": y, "is_imputed": False}, index=idx)
    p = pd.DataFrame(
        {
            "origin": idx[:-24],
            "target_time": idx[24:],
            "h": 24,
            "fold": "f1",
            "model": "ridge",
            "regime": "none",
            "y": y[24:],
            "yhat": y[24:] + 2,
            "block": "test",
            "target_imputed": False,
        }
    )
    folds = [{"name": "f1", "test_start": str(idx[0]), "test_end": str(idx[-1] + pd.Timedelta(hours=1))}]
    return p, obs, folds


def test_export_alignment_units_errors_and_metric_reproduction():
    p, obs, folds = source()
    out = evaluation_rows(p, obs, folds)
    assert len(out) == len(p)
    np.testing.assert_allclose(out.signed_error_mw, 2)
    np.testing.assert_allclose(out.absolute_error_mw, 2)
    published = pd.DataFrame(
        {"model": ["ridge"], "regime": ["none"], "h": [24], "n": [len(p)], "mae": [2], "rmse": [2]}
    )
    assert reproduce_metrics(out, published).iloc[0].mae_mw == pytest.approx(2)
    published.loc[0, "n"] -= 1
    with pytest.raises(ValueError):
        reproduce_metrics(out, published)


@pytest.mark.parametrize("fault", ["time", "duplicate", "observation", "timezone"])
def test_export_rejects_invalid_evidence(fault):
    p, obs, folds = source()
    if fault == "time":
        p.loc[0, "target_time"] += pd.Timedelta(hours=1)
    elif fault == "duplicate":
        p = pd.concat([p, p.iloc[:1]], ignore_index=True)
    elif fault == "observation":
        obs.loc[p.target_time.iloc[0], "demand"] += 100
    else:
        p["origin"] = p.origin.dt.tz_localize(None)
    with pytest.raises((ValueError, AssertionError)):
        evaluation_rows(p, obs, folds)


def test_complete_week_selection_is_deterministic_and_explicit():
    p, obs, folds = source()
    rows = evaluation_rows(p, obs, folds)
    a = select_weeks(rows, folds, ["ridge"], [24])
    b = select_weeks(rows.sample(frac=1, random_state=2), folds, ["ridge"], [24])
    pd.testing.assert_frame_equal(a, b)
    assert a.example.tolist() == ["good", "typical", "poor"]
    assert ((a.end - a.start) == pd.Timedelta(days=7)).all()
    assert a.start.is_unique


def test_plot_exports_valid_svg_without_trailing_whitespace(tmp_path):
    p, obs, folds = source()
    rows = evaluation_rows(p, obs, folds).iloc[:168]
    plot_week(rows, SimpleNamespace(example="test", fold="f1", selection_model="ridge"), tmp_path)
    svg = tmp_path / "actual_vs_predicted_test.svg"
    ElementTree.parse(svg)
    assert all(line == line.rstrip() for line in svg.read_text().splitlines())
    assert (tmp_path / "actual_vs_predicted_test.png").stat().st_size > 0

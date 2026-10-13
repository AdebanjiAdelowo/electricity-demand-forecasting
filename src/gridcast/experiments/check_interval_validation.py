"""Independent numeric and causal checks of a completed interval-selection run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridcast.data import dataset
from gridcast.experiments.interval_validation import BASE, DEV
from gridcast.experiments.revalidate import fingerprint


def check(out: Path):
    run = json.loads((out / "test_run.json").read_text())
    dev = Path(run["development_source"])
    selection = json.loads((out / "selection.json").read_text())
    assert run["status"] == "completed"
    assert run["peak_rss_bytes"] <= run["memory_limit_bytes"]
    assert run["wall_seconds"] <= run["time_limit_seconds"]
    assert fingerprint(dataset.PROCESSED) == run["processed_sha256"]
    assert fingerprint(out / "selection.json") == run["selection_sha256"]
    assert fingerprint(dev / "selection.json") == run["selection_sha256"]
    assert fingerprint(out / "protocol.json") == selection["protocol_sha256"]
    assert fingerprint(dev / "development_predictions.parquet") == selection["development_predictions_sha256"]
    for name, expected in run["baseline_prediction_sha256"].items():
        assert fingerprint(BASE / name) == expected
    candidates = pd.read_csv(dev / "development_candidates.csv")
    for policy in selection["policies"]:
        g = candidates[
            (candidates.model == policy["model"]) & (candidates.h == policy["h"]) & (candidates.alpha == 0.1)
        ]
        valid = g[(g.coverage >= 0.88) & (g.coverage <= 0.92) & g.complete]
        chosen = None if valid.empty else int(valid.sort_values(["winkler_mw", "days"]).iloc[0].days)
        assert chosen == policy["days"]
    development = pd.read_parquet(dev / "development_predictions.parquet")
    assert (development.origin < DEV.test_end).all()
    target_eligible = development[
        (development.block == "test") & ~development.target_imputed & (development.target_time < DEV.test_end)
    ]
    assert (target_eligible.target_time < pd.Timestamp("2024-04-01", tz="UTC")).all()
    frame = pd.read_parquet(out / "test_intervals.parquet")
    assert not frame.duplicated(["model", "regime", "fold", "h", "origin", "alpha", "method"]).any()
    assert (frame.target_time == frame.origin + pd.to_timedelta(frame.h, unit="h")).all()
    assert (~frame.target_imputed).all()
    selected = frame[frame.method == "selected_policy"]
    for policy in selection["policies"]:
        original = pd.read_parquet(BASE / f"{policy['model']}.parquet")
        g = selected[(selected.model == policy["model"]) & (selected.h == policy["h"])]
        reference = original.set_index(["fold", "origin", "h"])
        ix = pd.MultiIndex.from_frame(g[["fold", "origin", "h"]])
        np.testing.assert_array_equal(reference.reindex(ix).y.to_numpy(), g.y.to_numpy())
        np.testing.assert_array_equal(reference.reindex(ix).yhat.to_numpy(), g.yhat.to_numpy())
        if policy["days"] is None:
            continue
        known = g.last_feedback_at.notna()
        assert (g.loc[known, "last_feedback_at"] <= g.loc[known, "origin"]).all()
        for row in g.iloc[np.linspace(0, len(g) - 1, 24, dtype=int)].itertuples():
            p = original[(original.fold == row.fold) & (original.h == row.h)]
            eligible = p[
                (p.target_time <= row.origin)
                & (p.target_time > row.origin - pd.Timedelta(days=policy["days"]))
                & ~p.target_imputed
                & np.isfinite(p.y)
                & np.isfinite(p.yhat)
            ]
            scores = np.sort(abs(eligible.y - eligible.yhat).to_numpy())
            assert row.n_scores == len(scores)
            if len(scores) < 168:
                assert np.isnan(row.lo) and np.isnan(row.hi)
            else:
                rank = min(len(scores), int(np.ceil((len(scores) + 1) * (1 - row.alpha))))
                np.testing.assert_allclose(
                    [row.lo, row.hi],
                    [row.yhat - scores[rank - 1], row.yhat + scores[rank - 1]],
                    rtol=0,
                    atol=1e-8,
                )
    key = ["model", "regime", "fold", "h", "origin", "target_time", "alpha"]
    issued = selected[np.isfinite(selected.lo)].set_index(key)
    static = frame[frame.method == "conformal_static"].set_index(key).reindex(issued.index)
    matched = pd.concat([issued.reset_index(), static.reset_index()], ignore_index=True)
    for label, fields in [("overall", []), ("season", ["season"]), ("fold", ["fold"])]:
        table = pd.read_csv(out / f"test_{label}.csv")
        for row in table.itertuples():
            g = matched[
                (matched.model == row.model)
                & (matched.method == row.method)
                & (matched.h == row.h)
                & (matched.alpha == row.alpha)
            ]
            for field in fields:
                g = g[g[field] == getattr(row, field)]
            y, lo, hi = g.y.to_numpy(), g.lo.to_numpy(), g.hi.to_numpy()
            score = hi - lo + 2 / row.alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
            assert len(g) == row.n
            np.testing.assert_allclose(
                [((y >= lo) & (y <= hi)).mean(), (hi - lo).mean(), score.mean()],
                [row.coverage, row.width_mw, row.winkler_mw],
                rtol=1e-12,
                atol=1e-8,
            )
    for row in pd.read_csv(out / "test_availability.csv").itertuples():
        g = selected[(selected.model == row.model) & (selected.h == row.h) & (selected.alpha == row.alpha)]
        assert len(g) == row.n_targets
        assert g.lo.notna().sum() == row.n_issued
        assert g.lo.isna().sum() == row.n_unavailable
        np.testing.assert_allclose(((g.y >= g.lo) & (g.y <= g.hi)).mean(), row.coverage_missing_as_failure)
    print(
        "Verified frozen selection, resource limits, preserved archives, causal score samples, exact point predictions, 432 aggregate/slice rows and issuance denominators."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    check(parser.parse_args().output)

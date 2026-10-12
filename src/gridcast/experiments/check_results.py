"""Check completed evidence packages without training or network access."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.evaluation.backtest import scoreable
from gridcast.experiments.revalidate import fingerprint


def main() -> None:
    checked = 0
    for path in (config.RESULTS / "forecasting_baselines").glob("*/run.json"):
        record = json.loads(path.read_text())
        table = pd.read_csv(path.parent / "overall.csv")
        ends = {f["name"]: pd.Timestamp(f["test_end"]) for f in record["folds"]}
        for name, g in table.groupby("model"):
            p = scoreable(pd.read_parquet(path.parent / f"{name}.parquet"))
            p = p[p.target_time < p.fold.map(ends)]
            for row in g.itertuples():
                a = p[(p.h == row.h) & (p.regime == row.regime)]
                assert len(a) == row.n
                np.testing.assert_allclose(abs(a.y - a.yhat).mean(), row.mae, rtol=1e-10)
        checked += 1
    for path in (config.RESULTS / "weather_uncertainty").glob("*/run.json"):
        record = json.loads(path.read_text())
        for relative, expected in record["baseline_prediction_sha256"].items():
            assert fingerprint(config.ROOT / relative) == expected, relative
        intervals = pd.read_parquet(path.parent / "intervals.parquet", columns=[
            "model", "regime", "fold", "h", "origin", "target_time", "method", "alpha",
            "last_feedback_at", "y", "lo", "hi"])
        assert not intervals.duplicated(["model", "regime", "fold", "h", "origin", "method", "alpha"]).any()
        online = intervals[intervals.method != "conformal_static"]
        assert (online.last_feedback_at <= online.origin).all()
        assert (intervals.target_time == intervals.origin + pd.to_timedelta(intervals.h, unit="h")).all()
        table = pd.read_csv(path.parent / "interval_overall.csv")
        for row in table.itertuples():
            p = intervals[(intervals.model == row.model) & (intervals.method == row.method) &
                          (intervals.h == row.h) & (intervals.alpha == row.alpha)]
            assert len(p) == row.n
            np.testing.assert_allclose(((p.y >= p.lo) & (p.y <= p.hi)).mean(), row.coverage)
            np.testing.assert_allclose((p.hi - p.lo).mean(), row.width_mw)
        checked += 1
    sys.path.insert(0, str(config.ROOT))
    from experiments.physics_informed_synthetic_energy.network import solve

    for path in (config.RESULTS / "physics_informed_synthetic_energy").glob("*/run.json"):
        table = pd.read_csv(path.parent / "metrics.csv")
        for row in table.itertuples():
            with np.load(path.parent / f"{row.method}_seed{row.seed}.npz") as data:
                # Archived runs may use a different projection anchor; saved operating points remain checkable.
                state = solve(data["profiles"])
                np.testing.assert_allclose(state["feasible"].all(axis=1).mean(), row.trajectory_feasible_fraction)
                np.testing.assert_allclose(state["residual"], data["residual"], atol=1e-10)
                if row.method == "projected":
                    assert state["feasible"].all()
        checked += 1
    print(f"Checked {checked} completed evidence packages" if checked else "No completed evidence packages present")


if __name__ == "__main__":
    main()

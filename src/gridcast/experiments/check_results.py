"""Check completed evidence packages without training or network access."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.evaluation.backtest import scoreable


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

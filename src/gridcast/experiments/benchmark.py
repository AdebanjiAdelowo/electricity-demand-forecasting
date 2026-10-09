"""Main rolling-origin benchmark: statistical vs machine-learning vs deep models.

Each model is refit on every fold and produces forecasts for every hourly
origin of the fold's calibration and test blocks, for every horizon in the
dense grid. Predictions are stored per model; `report.py` turns them into
tables.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from gridcast import config, tracking
from gridcast.data import dataset
from gridcast.evaluation import backtest
from gridcast.experiments import zoo

EXPERIMENT = "benchmark"


def run(models: list[str] | None = None, experiment: str = EXPERIMENT, folds=None, horizons=None,
        skip_existing: bool = True) -> pd.DataFrame:
    config.ensure_dirs()
    df = dataset.load()
    folds = folds or config.TEST_FOLDS
    horizons = horizons or config.DENSE_HORIZONS
    registry = {**zoo.all_models(), **zoo.external()}
    names = models or list(registry)
    timing_path = config.TABLES / f"{experiment}_timing.json"
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    with tracking.Run(experiment, {"models": names, "horizons": horizons}, folds) as run_:
        for name in names:
            out = config.PREDICTIONS / experiment / f"{name}.parquet"
            if skip_existing and out.exists() and name in timings:
                print(f"skip {name} (exists)")
                continue
            print(f"[{experiment}] {name}", flush=True)
            h = [24] if name == "ercot_day_ahead_reference" else horizons
            pred, timing = backtest.run_backtest(registry[name], df, folds, h)
            run_.artifact(backtest.save_predictions(pred, experiment, name))
            timing["hardware"] = tracking.hardware()
            timings[name] = timing
            timing_path.write_text(json.dumps(timings, indent=2, default=str))
            s = backtest.scoreable(pred)
            s = s[s["h"].isin(config.HEADLINE_HORIZONS)]
            mae = s.assign(ae=(s.y - s.yhat).abs()).groupby(["regime", "h"])["ae"].mean().round(0)
            print(mae.unstack("h").to_string(), flush=True)
            run_.log(**{f"{name}/mae": {f"{r}|{hh}": float(v) for (r, hh), v in mae.items()}})
    return pd.DataFrame(timings).T


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--group", choices=["classical", "deep", "external", "all"], default=None)
    ap.add_argument("--fold-set", choices=["test", "dev", "uri"], default="test")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    models = a.models
    if a.group and not models:
        models = list({"classical": zoo.classical, "deep": zoo.deep, "external": zoo.external,
                       "all": lambda: {**zoo.all_models(), **zoo.external()}}[a.group]())
    folds = {"test": config.TEST_FOLDS, "dev": [config.DEV_FOLD], "uri": [config.URI_FOLD]}[a.fold_set]
    exp = EXPERIMENT if a.fold_set == "test" else f"{EXPERIMENT}_{a.fold_set}"
    run(models, exp, folds, skip_existing=not a.force)


if __name__ == "__main__":
    main()

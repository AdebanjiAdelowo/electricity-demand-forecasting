"""Prepared CPU/MPS/Kaggle/Colab runner. Execution requires separate approval.

This script installs nothing and downloads nothing. Supply the repository,
processed dataset and project-specific locked environment on the target host.
No GPU execution is implied by the existence of this runner.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from gridcast import config, tracking
from gridcast.data import dataset
from gridcast.evaluation import backtest
from gridcast.experiments.revalidate import fingerprint
from gridcast.models.deep import DeepForecaster, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["tcn", "gru", "nbeats"], default="tcn")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("Positive training budget required")
    if shutil.disk_usage(config.ROOT).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free; leave the experiment pending")
    # Caller must additionally check competing jobs and memory on the target host.
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    settings = TrainConfig(arch=args.arch, weather=False, hidden=32, layers=2, device=args.device,
                           seed=args.seed, epochs=args.epochs, patience=3)
    out = config.RESULTS / "neural_benchmarks" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    models = []

    def factory():
        model = DeepForecaster(settings, f"{args.arch}_none")
        models.append(model)
        return model

    df = dataset.load()
    pred, timing = backtest.run_backtest(factory, df, config.TEST_FOLDS, config.HEADLINE_HORIZONS)
    pred.to_parquet(out / "predictions.parquet", index=False)
    for fold, model in zip(config.TEST_FOLDS, models, strict=True):
        torch.save(model.net.state_dict(), out / f"{fold.name}.pt")
    (out / "run.json").write_text(json.dumps({"git": tracking.git_state(), "config": vars(settings),
                                            "device": args.device, "versions": tracking.versions(),
                                            "dataset_sha256": fingerprint(dataset.PROCESSED), "timing": timing,
                                            "folds": [tracking.fold_dict(f) for f in config.TEST_FOLDS]}, indent=2))
    print(Path(out).resolve())


if __name__ == "__main__":
    main()

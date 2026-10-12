"""Run the GPU experiments of this study on Kaggle (T4) or any CUDA machine.

Usage on Kaggle
    1. Zip the repository (`git archive -o gridcast.zip HEAD`) and upload it as a
       private Kaggle dataset. The processed dataset is part of the repository,
       so no download from EIA or Open-Meteo is needed.
    2. Provision a project-specific locked environment only after approval.
    3. Inspect the plan with --src; execution additionally requires --execute
       and --python pointing to the prepared project environment's interpreter.
    4. Preserve downloaded outputs in a separate directory, never over old results.

This legacy multi-stage runner installs nothing. It is pending GPU execution,
and its weather stages retain unverified availability assumptions. The newer
scripts/neural_benchmark_pending.py is the preferred no-weather comparison.
Run metadata may have a null commit in a git archive; retain the original
archive's commit and fingerprints externally. Choose stages explicitly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

STAGES = {
    # name: (description, command arguments for `python -m gridcast.cli`)
    "deep_benchmark": (
        "GRU, TCN and N-BEATS on the four test folds, 20 epochs each",
        ["benchmark", "--group", "deep", "--force"],
    ),
    "size_gru": ("GRU hidden-size study on the dev fold", ["study", "size", "--arch", "gru"]),
    "size_tcn": ("TCN channel-width study on the dev fold", ["study", "size", "--arch", "tcn"]),
    "lookback_gru": ("GRU lookback study on the dev fold", ["study", "lookback", "--arch", "gru"]),
    "seeds_nbeats": (
        "five-seed variation of N-BEATS on the dev fold",
        ["study", "seeds", "--arch", "nbeats"],
    ),
    "seeds_gru": ("five-seed variation of the GRU on the dev fold", ["study", "seeds", "--arch", "gru"]),
    "report": ("rebuild tables after the deep benchmark", ["report"]),
    "probabilistic": ("recompute calibration tables including deep models", ["probabilistic"]),
}
DEFAULT = [
    "deep_benchmark",
    "size_gru",
    "size_tcn",
    "lookback_gru",
    "seeds_nbeats",
    "seeds_gru",
    "report",
    "probabilistic",
]


def sh(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        env={**os.environ, "GRIDCAST_ROOT": str(cwd), "PYTHONPATH": str(cwd / "src")},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding the repository (read-only is fine)")
    ap.add_argument("--work", default="/kaggle/working/gridcast")
    ap.add_argument("--stages", nargs="+", default=["deep_benchmark"], choices=list(STAGES))
    ap.add_argument("--execute", action="store_true", help="run only after explicit compute approval")
    ap.add_argument("--python", help="interpreter in a separately prepared project virtual environment")
    a = ap.parse_args()
    if not a.execute:
        for name in a.stages:
            print(f"PENDING: {name}: {STAGES[name][0]}")
        return
    if not a.python or not Path(a.python).is_file():
        raise ValueError(
            "Execution requires an existing project environment interpreter; no automatic install"
        )
    work = Path(a.work)
    if work.exists():
        raise ValueError("Use a fresh work directory; preserve previous outputs")
    shutil.copytree(a.src, work, ignore=shutil.ignore_patterns(".venv", ".git", "results", "__pycache__"))
    sh(
        [a.python, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable; leave pending'"],
        work,
    )
    for name in a.stages:
        desc, args = STAGES[name]
        print(f"\n=== {name}: {desc} ===", flush=True)
        sh([a.python, "-m", "gridcast.cli", *args], work)
    out = shutil.make_archive(
        "/kaggle/working/gridcast_gpu_results"
        if Path("/kaggle/working").exists()
        else str(work.parent / "gridcast_gpu_results"),
        "zip",
        work / "results",
        base_dir=".",
    )
    print("wrote", out)


if __name__ == "__main__":
    main()

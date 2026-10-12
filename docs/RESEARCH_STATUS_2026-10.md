# Research audit, 12 October 2026

## Inspection baseline

HEAD at inspection: `b1c92e3` (preserved). Existing modifications in
`src/gridcast/experiments/studies.py`, and untracked `kaggle/`, `results/`,
`copilot_eval.py`, `viz/plots.py`, and `test_anomaly.py` predate this work.
No historical evidence is to be overwritten. README was empty; docs,
configs and scripts directories were empty. The project has a uv lockfile
and an existing 1.2 GiB environment. Approximately 9.6 GiB disk was free.
Process inspection was denied by the sandbox, so concurrent training cannot
be ruled out. This study uses CPU only and bounded thread counts.

## Component classification at inspection

| Component | Classification | Evidence and limits |
|---|---|---|
| EIA-930 demand ingestion and causal cleaning | Implemented but not scientifically evaluated | Workbook and manifest available; validation must recheck raw-to-processed consistency and fingerprints |
| UTC grid and local calendar/DST | Implemented and validated | Explicit UTC hourly grid and DST unit tests; source revisions remain possible |
| Lag/calendar features and train-only scaler | Implemented and validated | Future-demand perturbation tests; target-aware training mask; Ridge pipeline fits scaler only on training subset |
| Seasonal/persistence, Ridge, LightGBM, SARIMAX, double seasonal HW | Implemented but not scientifically evaluated | Stored four-fold predictions and tables exist; independent scoring and source consistency audit still needed |
| Neural GRU/TCN/N-BEATS-style forecasting | Implemented but not scientifically evaluated | Code and shape tests exist, no completed benchmark artifacts found |
| Conformal and quantile intervals | Implemented but not scientifically evaluated | Static/rolling calibration code exists; empirical coverage must be evaluated separately by information regime |
| Anomaly detection and injection | Implemented but not scientifically evaluated | Detector and synthetic-injection studies; no real labelled event validation found |
| Feature ablation, robustness, Uri, neural size/seed studies | Partially implemented | Runners exist; no completed evidence found |
| API/dashboard/copilot | Partially implemented | API/tests and grounded tools exist; README absent; CLI/CI command audit needed |
| Physics/network simulation and generative models | Planned but not implemented | No existing network/generative implementation found |
| Application research summary | Planned but not implemented | No summary or research reports found |

## Scientific risks

* Weather oracle uses future reanalysis and cannot be presented as operational forecasting.
* The 168-hour archived-weather bucket is capped at seven days and falls short
  of the assumed six-hour publication delay. Do not use it for operational claims.
* Forecast weather issue-time assumptions need verification against provider
  archives; training uses reanalysis, producing a distribution shift.
* Historical run metadata names the 9 October dataset; current manifest names
  10 October. Raw EIA hash agrees, but no historical processed-file hash exists.
  Full identity of historical weather inputs cannot be established retrospectively.
* Existing overall MASE uses a common pre-test scale rather than fold-specific
  scaling. New scoring should state the exact scaling convention.
* Existing probabilistic report groups by model/method/horizon and pools
  weather regimes. New tables must include regime.
* Existing CI refers to absent `gridcast.experiments.check_results`.
* `gridcast` was not importable from the initial environment without
  `PYTHONPATH=src`; reinstalling the local package is required.
* Historical run hardware lists available MPS, which is not evidence that any
  neural benchmark actually ran on MPS.

## Integration plan

Reuse dataset loading, fold definitions, forecasting models, metrics and
conformal calibration. Add separate, append-only output directories for
revalidated forecasting and one small simulated network experiment.
Use a simple probabilistic window generator with explicit post-generation
projection. This changes generated samples, not the learned objective.
Do not claim diffusion, GNN, privacy, field deployment or end-to-end
physics-informed training.

## README constraint

The parent `AGENTS.md` requests `github-readme-standardizer` for substantial
README work. That skill was not found in the available skill directories or
plugin cache. The user subsequently explicitly authorized direct README
authoring, following the same project-focused policy.

## Completed evidence after the initial audit

The no-weather five-model forecasting benchmark was rerun on four test folds
in `results/forecasting_baselines/20261012T093203Z/`. Raw EIA cleaning exactly
reconstructs all processed demand values and flags. Predictions, fold/season/
demand slices, static conformal scores and paired bootstrap intervals are
available. These five models now have implemented and validated computational
evaluation under this protocol; weather/neural/stress claims remain outside it.

The five-bus generator study completed in
`results/physics_informed_synthetic_energy/20261012T093721Z/`, with independent
pandapower checks, three sampling seeds, box-only ablation, fidelity/diversity/
feasibility evaluation, downstream simulated forecasting and figures. It is
implemented and validated as a bounded computational proof of concept, not a
validated real-system generative model. A prior run is preserved; the final
run uses a zero-generation correction anchor. Numerical reports and the
application summary cite the final run.

Added a meaningful `check_results` module for the previously broken CI command,
and extended CI to install optional physics dependencies and lint experiments.
The optional pandapower dependency required NumPy 2.4.6 and pandas 2.3.3,
replacing the environment's newer versions; uv.lock records the resolution.
No global packages, other repositories or shared caches were changed.

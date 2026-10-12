# GridCast

Electricity-demand forecasting on hourly ERCOT data, with chronological
backtesting, quantile forecasts, conformal intervals and a small experiment
on physically feasible synthetic energy time series.

## Forecasting

The project processes EIA-930 TEX hourly demand in MW on an explicit UTC grid.
Calendar features use America/Chicago local time and account for daylight
saving transitions. Causal demand cleaning flags repaired targets for
exclusion from evaluation. Forecast features use observed demand through the
origin, deterministic calendar information and separately labelled weather
information regimes.

Implemented models include persistence, daily and weekly seasonal naive,
seasonal profiles, Ridge, LightGBM, double seasonal Holt–Winters, SARIMAX,
and optional PyTorch GRU, TCN and N-BEATS-style networks. The core comparison
below uses five existing models without weather, evaluated on the same
four expanding test folds covering July 2024 through June 2026.

| Model | 1h MAE (MW) | 24h MAE (MW) | 48h MAE (MW) | 168h MAE (MW) |
|---|---:|---:|---:|---:|
| Persistence | 1551.89 | 2443.70 | 3593.78 | 4802.12 |
| Daily seasonal naive | 2446.13 | 2443.70 | 3593.78 | 4802.12 |
| Weekly seasonal naive | 4786.23 | 4796.07 | 4810.00 | 4802.12 |
| Ridge | 313.17 | 2234.26 | 3194.42 | 4136.95 |
| LightGBM | 329.67 | 2319.03 | 3367.39 | 4088.59 |

The [forecasting report](docs/FORECASTING_BENCHMARK_REPORT.md) explains
splits, fingerprints, metric definitions, uncertainty intervals and seasonal
error slices. Static conformal intervals undercovered at longer horizons;
nominal coverage is not guaranteed for changing, dependent demand.

The [weather and uncertainty study](docs/WEATHER_UNCERTAINTY_REPORT.md) adds
explicit forecast-publication gating, labelled historical/oracle weather
controls, residual-drift diagnostics and timestamp-aware online calibration.
At 168 hours, rolling calibration raised Ridge's measured 90% coverage from
82.81% to 88.62% with wider intervals. The adaptive variant performed worse
at long horizons. The validated no-weather point benchmark remains primary.

```sh
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLCONFIGDIR=results/scratch/matplotlib \
  .venv/bin/python -m gridcast.experiments.weather_uncertainty
```

## Synthetic energy experiment

The [five-bus experiment](experiments/physics_informed_synthetic_energy)
fits a Gaussian PCA model to simulated daily load and renewable windows.
It compares identical raw samples, bound clipping and an explicit
post-generation AC feasibility correction. A balanced radial solver supplies
voltages, current loading and nodal residuals; selected states are checked
independently with pandapower.

Across three sampling seeds, whole-day operating feasibility rose from 85.68%
after bound clipping to 100% after network correction. Correction reduced
diversity and did not improve the held-out simulated forecasting proxy.
This is a small simulated-network experiment with post-processing, not
physics-informed neural training or real operational energy data.
The [scientific report](docs/PHYSICS_INFORMED_SYNTHETIC_ENERGY_REPORT.md)
contains equations, quantitative comparisons, figures and limitations.

## Installation

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are required.
Use the repository-specific environment and lockfile:

```sh
uv sync --locked --extra dev --extra physics
```

Optional extras are `deep` (PyTorch), `serve` (API/dashboard), and `copilot`.
Add only the extras needed for the chosen task. No CUDA-only package is
required by the default environment. The documented experiment uses CPU.

## Data preparation

The fixed processed snapshot is `data/processed/ercot_hourly.parquet`, with
provenance in `data/processed/manifest.json`. The independent benchmark also
requires `data/raw/Region_TEX.xlsx` to reconstruct and verify demand cleaning.
Keep raw files outside Git. Obtain them through the implemented source loader:

```sh
# Downloads public demand and weather; rebuilds the processed snapshot.
PYTHONPATH=src .venv/bin/python -m gridcast.cli data
```

This rebuild command changes the dataset fingerprint. Retain an existing
snapshot when reproducing published numbers. Demand comes from the
[EIA Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/).
Weather attribution and source URLs are recorded in the manifest.
No real forecasting data are replaced with synthetic observations.

## Reproduce the experiments

Run from the repository root:

```sh
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLCONFIGDIR=results/scratch/matplotlib \
  .venv/bin/python -m gridcast.experiments.revalidate

PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLCONFIGDIR=results/scratch/matplotlib \
  .venv/bin/python -m experiments.physics_informed_synthetic_energy.run

PYTHONPATH=src .venv/bin/python -m gridcast.experiments.check_results
```

Each runner creates a new timestamped results directory. It saves configuration,
random seeds, package versions, source and dataset hashes, device, timing,
predictions or network states, and quantitative tables. The physics runner
also saves PNG/SVG figures and captions. Generated outputs are excluded from
Git; the reports retain measured results and identify their local evidence.

## Tests

For all optional test paths, install the additional dependencies:

```sh
uv sync --locked --extra dev --extra physics --extra deep --extra serve --extra copilot
make test
make lint
```

Tests use deterministic synthetic fixtures, CPU and no network. Neural and
other tests run in separate Python processes because the combined PyTorch/
LightGBM test process stalled on the tested macOS environment. The suite
checks temporal leakage, DST, training boundaries, forecasting behaviour,
metrics, physical equations, operating limits, independent solver agreement,
generator shapes, deterministic sampling and failure handling.

## Limitations and research directions

ERCOT observations are revised historical data, without point-in-time revision
archives or telemetry publication-delay modelling. Future observed-weather
oracle results cannot be treated as operational forecasts. Seven-day forecast
weather also falls short of the assumed publication delay at the 168-hour
horizon. The completed no-weather comparison avoids those weather assumptions;
full neural benchmark results are not reported here.

The illustrative feeder does not represent a deployed grid, unbalanced
three-phase behaviour or network contingencies. Its reference process rejects
infeasible simulated days. Gaussian generation does not enforce daylight
renewable availability. Correction improves the stated constraints without
establishing privacy, real-data fidelity or operational utility. Useful next
steps include a published feeder, resource-aware renewable bounds, independent
training replications and a matched training-penalty experiment.

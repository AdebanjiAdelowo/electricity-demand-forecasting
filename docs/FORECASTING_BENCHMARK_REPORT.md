# ERCOT forecasting benchmark, 12 October 2026

## Scope and provenance

This independently executed comparison reuses the repository's persistence,
daily/weekly seasonal naive, Ridge and LightGBM implementations. It evaluates
the **no-weather** information regime. Forecast/oracle weather measurements
in older outputs are retained but excluded from the primary claims here.

Source: [EIA Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/),
Form EIA-930 regional TEX workbook, hourly electricity demand in MW. EIA
describes these submissions as preliminary; this is a fixed retrospective
snapshot, not a point-in-time archive of revisions. Demand licensing is
recorded as US government public-domain work in the manifest. Weather from
Open-Meteo remains in the processed file but is not used by this comparison.

Processed version `eia930-TEX+openmeteo@2026-10-10` contains 97,914 rows,
2015-07-01 06:00 UTC through 2026-08-31 23:00 UTC. A complete re-read and
cleaning of the raw workbook reproduces **all processed demand values and
imputation flags**. The raw source has zero duplicate timestamps, zero missing
grid timestamps and 72 missing demand observations; all are causally imputed,
with a longest gap of 24 hours. There are no remaining missing demand values.
Imputed targets are excluded from evaluation, but causal repaired history is
available to forecasters.

* Raw EIA SHA-256: `ff4f90c64f92a78910a18ee1824bf2ab7b05c7404b1bdab43a3db5cb6af34073`.
* Processed SHA-256: `1e3acfa6c4070d0dc69d6a2d854964f09afc42b3291221fda646c606249ca5ab`.
* Manifest SHA-256: `b881107165298776a96218f88ce60123340f049b960efcd03ebc8c555f1e34de`.

The UTC grid is ordered, unique and hourly. Converting it to naive Chicago
local time yields 11 repeated autumn DST hours; these remain distinct UTC
observations. Calendar features label the local start of each hour-ending
interval. Spring/fall transitions are covered by existing unit tests.

## Protocol and leakage controls

Four expanding-window folds have test-origin intervals

| Fold | Test origins (UTC, start inclusive/end exclusive) |
|---|---|
| test1 | 2024-07-01 to 2025-01-01 |
| test2 | 2025-01-01 to 2025-07-01 |
| test3 | 2025-07-01 to 2026-01-01 |
| test4 | 2026-01-01 to 2026-07-01 |

Each fold reserves 91 days for validation and 91 days for calibration before
its test interval; training starts 2015-08-01. Exact timestamps are in
`run.json`. Horizons are 1, 24, 48 and 168 hours. Both origin and target must
lie within a fold's test interval. This removes cross-boundary targets and
explains differences from historical headline tables. Models are refit per
fold; forecasts issue at every hour and may use demand through the origin.

All five models share target keys and input demand availability. Calendar
features are deterministic. Demand lags and trailing normalization use no
future values. Supervised training masks require the target to precede the
fit cutoff. Ridge selects its penalty from the validation period and refits
its StandardScaler and estimator on fit+validation only. LightGBM uses its
existing fixed hyperparameters and quantile fits on fit+validation. Calibration
residuals are admitted only when their targets precede test start. Static
absolute-residual conformal intervals use those fold-specific residuals.

The current revalidated data are checked independently against raw EIA demand.
No synthetic data were substituted. The revision history of EIA observations
is not reconstructed, and source-reporting delays are not modelled.

## Metrics and uncertainty definitions

MAE and RMSE are in MW. sMAPE is

\[
100\,\operatorname{mean}\left[\frac{2|y-\hat y|}{|y|+|\hat y|}\right].
\]

MASE divides each error by its fold's training+validation seasonal-naive
denominator, the mean of \(|y_t-y_{t-168}|\) on the fitting series, then
averages scaled errors. It includes causally imputed training observations.
No test information enters this denominator. Pinball loss is averaged over
the repository's nine quantiles; interval coverage is the fraction of targets
between endpoints, and width is the mean endpoint difference in MW.

Paired MAE differences to weekly seasonal naive use common origin/horizon
keys. The 95% moving-block bootstrap has 500 replicates, seven-day blocks
within each fold, seed 0, and valid-hour weighting. Blocks never cross fold
boundaries. These intervals reflect temporal sampling variation conditional
on the fitted models, not hyperparameter or dataset uncertainty. Dependence
beyond a week may be underrepresented, particularly at the 168-hour horizon.

## Validated numerical results

Executed evidence: `results/forecasting_baselines/20261012T093203Z/`.

| Model | 1h MAE | 24h MAE | 48h MAE | 168h MAE |
|---|---:|---:|---:|---:|
| Persistence | 1551.89 | 2443.70 | 3593.78 | 4802.12 |
| Daily seasonal naive | 2446.13 | 2443.70 | 3593.78 | 4802.12 |
| Weekly seasonal naive | 4786.23 | 4796.07 | 4810.00 | 4802.12 |
| Ridge | **313.17** | **2234.26** | **3194.42** | 4136.95 |
| LightGBM | 329.67 | 2319.03 | 3367.39 | **4088.59** |

All entries are MW. Counts per model are 17,492, 17,400, 17,304 and 16,824,
respectively. Daily seasonal naive equals persistence at these daily-multiple
horizons because the latest observed same-hour reference is the origin.

| Model/horizon | RMSE (MW) | sMAPE (%) | MASE |
|---|---:|---:|---:|
| Ridge 1h | 426.61 | 0.5690 | 0.07435 |
| Ridge 24h | 3155.35 | 4.0412 | 0.53033 |
| Ridge 48h | 4364.90 | 5.7695 | 0.75838 |
| LightGBM 168h | 5467.26 | 7.3421 | 0.97151 |

Ridge improves 24-hour MAE by approximately 8.57% relative to daily naive.
LightGBM improves 168-hour MAE by approximately 14.86% relative to weekly
naive. These are measured aggregate comparisons, not universal model rankings.
Ridge 24-hour MAE difference to weekly naive is −2561.81 MW with a paired
95% interval [−3019.73, −2218.45]. LightGBM 168-hour difference is −713.53 MW
with interval [−1175.75, −336.66]. No claim of significance between Ridge and
LightGBM follows from these baseline-relative intervals.

## Uncertainty evaluation

| Model/horizon | Static conformal 90% coverage | Width (MW) | Pinball loss (MW) |
|---|---:|---:|---:|
| Ridge 1h | 90.33% | 1384.40 | 80.67 |
| Ridge 24h | 87.64% | 9301.34 | 599.87 |
| Ridge 48h | 86.26% | 11907.01 | 836.65 |
| Ridge 168h | 82.81% | 14351.87 | 1083.29 |
| LightGBM 1h | 89.93% | 1415.25 | 84.28 |
| LightGBM 24h | 88.22% | 9506.73 | 613.99 |
| LightGBM 48h | 86.73% | 12868.66 | 868.84 |
| LightGBM 168h | 86.41% | 15586.84 | 1049.10 |

These are empirical measurements, not guaranteed coverage for dependent,
nonstationary demand. LightGBM's raw 90% interval coverage is 88.84%, 85.44%,
83.66% and 79.76% across the four horizons. Static conformal calibration
improves long-horizon coverage here but does not attain the nominal rate.
Raw quantile pinball losses are in `overall.csv`; calibrated losses are in
`conformal_static.csv`. The absolute-residual construction and quantile grid
are not a calibrated full predictive density.

## Error analysis

`by_fold.csv`, `by_season.csv` and `by_demand_regime.csv` contain MAE, RMSE,
sMAPE and fold-scaled MASE slices. Season uses target-hour local time. Demand
regimes use pooled realized target quantiles (33%, 67%, 95%) for retrospective
diagnostics, not forecast inputs or a deployable classifier. Increasing
horizon increases aggregate error; rankings vary with horizon. High-load and
seasonal slices should be assessed with their sample counts; this report does
not infer stable operational performance from a pooled score alone.

For Ridge at 24 hours, seasonal MAE is 1871.96 MW in autumn (4370 targets),
2357.90 MW in spring (4414), 2038.78 MW in summer (4368), and 2679.49 MW in
winter (4248). Retrospective demand-regime MAE is 1910.79 MW for low demand,
2098.36 MW for middle demand, 2819.97 MW for high demand and 1973.35 MW for
the top 5% demand slice (868 targets). The top-demand slice is easier than
the broader high-demand slice in this particular evaluation; extremes must
not automatically be described as the worst-performing regime.

## Reproducibility and remaining experiments

```sh
uv sync --locked --extra dev --extra physics
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLCONFIGDIR=results/scratch/matplotlib \
  .venv/bin/python -m gridcast.experiments.revalidate
PYTHONPATH=src .venv/bin/python -m gridcast.experiments.check_results
```

The runner creates a new timestamped directory and retains all historical
outputs. Raw data must already exist; documented acquisition is implemented
by `gridcast data`, which downloads EIA and Open-Meteo and rebuilds processed
data. Do not invoke rebuilding merely to rerun this fixed snapshot.

The CPU run took 273.62 s, with two requested LightGBM/BLAS threads. Exact
versions, source hashes, lock hash, split definitions, device and per-model
training/inference timings are in `run.json`. Recorded HEAD is
`b1c92e37cbea546ca6bdd0eb8932f2c5cfd93795` with a dirty working tree; source
hashes, not HEAD alone, identify the executed implementation. The run preceded
the evidence-package commit. All stored predictions are full precision.

Existing SARIMAX, double seasonal HW and weather-aware outputs remain
historical evidence, not fresh validations. Existing neural forecasts have
unit/smoke coverage but no completed real-data benchmark evidence. Full neural
training is **PENDING**, avoiding accelerator work on the shared machine:

```sh
# After checking shared resources; this is an unexecuted full benchmark.
PYTHONPATH=src GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 \
  .venv/bin/python -m gridcast.experiments.benchmark --group deep
```

Before that command, resolve operational weather issue-time assumptions and
decide whether to evaluate neural models without weather for fair comparison.
MPS is available hardware, but no MPS or Kaggle benchmark is claimed here.
The bounded five-model no-weather comparison is the completed primary study.

# Weather and uncertainty extension: baseline audit

Inspected 12 October 2026 at HEAD `e6a1a6d9f6b978dc8fbc8c021df5a2482f6727e1`.
The previously validated four-fold, 1/24/48/168-hour no-weather protocol and
its immutable predictions remain the primary comparison. The processed EIA
snapshot, chronological boundaries, train-only Ridge scaling and seasonal
MASE definitions are documented in `FORECASTING_BENCHMARK_REPORT.md`.

Architecture separates ingestion, features, models, backtesting, uncertainty,
experiments and reporting. Existing forecasts include naive models, Ridge,
LightGBM, SARIMAX and double seasonal HW. GRU/TCN/N-BEATS-style modules have
CPU fixture tests, but there are no completed validated neural ERCOT runs.
The five-bus synthetic study is independent of this forecasting extension.

Baseline tests were run before code changes: 57 non-neural tests and 5 neural
tests passed in separate CPU processes (42.38 s and 19.49 s). Existing pandas
downcasting warnings remain. No package upgrade is needed. At inspection,
approximately 15 GiB disk was available and memory_pressure reported 34%
system-wide free memory. No GPU or long-running training is authorized.

## Information availability findings

* The historical weather file is reanalysis, an observation proxy with no
  point-in-time publication record. Future reanalysis is explicitly an oracle.
  Even past reanalysis requires a declared availability-delay assumption.
* Previous Runs weather retains valid-time lead-day buckets, not per-record
  verified issue and publication timestamps. Its metadata records download
  time, which cannot substitute for original availability time.
* The capped seven-day bucket cannot meet a six-hour release delay at 168h.
  Existing forecast code must not be called operationally verified merely
  because its demand features are causal.
* The existing rolling conformal method shifts by a number of rows equal to
  horizon hours. This relies on regular hourly origins; it is unsafe for
  irregular/missing forecasts. Its finite-sample level uses configured window
  size rather than actual valid score count.
* Static calibration and historical results should be preserved. New online
  evaluation should use explicit feedback timestamps and finite-score counts,
  with independently labelled methods and conditional coverage diagnostics.

Open-Meteo describes the [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
as fixed lead offsets. The [Single Runs product description](https://open-meteo.com/en/docs/historical-forecast-api)
allows initialization-time selection; verifying original publication time is
still a separate requirement. No additional archive download is performed here.

## Scope

Add a strict timestamp-gated feature path and retrospective history/oracle
controls using the existing Ridge estimator. Analyze existing validated point
forecasts and add timestamp-aware rolling and delayed adaptive calibration.
Save outputs in new directories and report negative results. Long neural/GPU
jobs remain pending explicit approval.

Pre-existing changes in `studies.py`, `kaggle/`, historical `results/`,
`copilot_eval.py`, `viz/plots.py` and `test_anomaly.py` will be preserved and
excluded from new commits. Historical reports and benchmark artifacts are not
to be overwritten.

# Synchronization, row-level evidence and next-stage review

## Repository inspection

Inspected `main`, tracking `origin/main`, at local tip `e40eec0`. The remote
was `e6a1a6d`; neither `22b59b2` nor `e40eec0` was present there. Divergence
was zero remote-only and two local-only commits, allowing a normal fast-forward.
The existing remote is `https://github.com/AdebanjiAdelowo/electricity-demand-forecasting.git`.

Remaining local work consisted of architecture-specific study dispatch,
anomaly tests, legacy plotting/copilot evaluation code, a Kaggle wrapper and
generated result families. All were reviewed before staging. The existing
code contributions are retained; narrowly justified safety/label fixes and
formatting were applied to the files being published.

Full predictions, table outputs, run archives, figures and serving caches remain
on disk but are now explicitly ignored. No deletion, stash, reset, history
rewrite or dataset modification was performed. Source/configuration/tests,
documentation and one reviewed small prediction-export bundle are suitable
for committing. Virtual environments, raw data, caches and checkpoints are
excluded. A complete local inventory was written to
`/private/tmp/gridcast-sync-inventory.json` (37,872 ignored files at inventory
time); it is not a public artifact and includes no file contents.

## Justified source fixes

* The legacy Kaggle wrapper no longer installs packages automatically. Its
  default action prints a pending plan without copying data or invoking a
  subprocess. Execution requires explicit approval, an existing project
  environment interpreter and a fresh working directory. No CUDA run occurred.
  Source/archive commit provenance must be retained externally when `.git`
  is absent. The dedicated no-weather neural runner remains preferred.
* Legacy plot comparisons labelled operational now include only `none`.
  Weather comparisons explicitly mark archived NWP availability as unverified
  and reanalysis as oracle. Missing plot categories are handled, and causal
  weather-event assertions were removed from an illustrative title.
* Copilot evaluation defaults to offline. Explicit LLM-only selection without
  credentials reports no execution instead of concatenating nonexistent scores.
  Offline harness routing is tested with fixtures; no paid API was called.
* Existing architecture-specific size/lookback/seed studies and anomaly tests
  are retained and included in the source publication. Their presence does not
  establish completed real-data neural/anomaly studies.

## Saved prediction versus ground-truth verification

The validated parquet files contain row-level issue timestamps, target
timestamps, horizons, folds, model names, `y` and `yhat`. The export uses those
saved values and independently joins `y` against processed **and raw** EIA
demand at target timestamps. Only non-imputed evaluation targets within each
test fold are eligible. UTC alignment, positive horizons, duplicate keys,
finite values and published row counts are checked.

`src/gridcast/experiments/export_predictions.py` refits no model and regenerates
no forecasts. It reproduced all **20** MAE/RMSE table rows (five models × four
horizons) to numerical tolerance before selecting examples. Signed error is
predicted minus observed MW; absolute error is its magnitude. CSV round trips
are checked, and original prediction hashes are verified again afterward.

Reviewed output: `results/prediction_exports/20261012T163902Z/`.
The three CSVs each contain 1,344 rows: two models × four horizons × 168
target hours. All nine requested fields are present; there are no missing
values. Full input archives are not published. The entire reviewed bundle is
about 1.5 MiB; each individual file is under 300 KiB, including figures. This
size/purpose review supports publishing the compact evidence rather than
large parquet archives. Source hashes, data fingerprint, export provenance,
row counts, selection method and artifact sizes are recorded in the manifest.

| Example | Fold | Target period, UTC (end exclusive) | Ridge 24h MAE (MW) |
|---|---|---|---:|
| Good | test1 | 2024-08-12 to 2024-08-19 | 903.71 |
| Median-ranked | test1 | 2024-07-29 to 2024-08-05 | 2002.78 |
| Poor | test2 | 2025-02-19 to 2025-02-26 | 7036.82 |

These are deliberately post-hoc illustrations selected by minimum, median-
ranked and maximum Ridge 24h error from 97 complete seven-day common windows.
They are not unbiased aggregate performance estimates. Plots show the actual
load, both 24-hour forecasts and signed errors, with UTC/units/legends; they
were visually inspected. CSVs contain all four horizons.

## Checks and formatting limitation

The documented baseline test workflow initially passed 100 tests. Eleven new
tests cover export alignment, exact metric reproduction, invalid evidence,
deterministic week selection, safe pending execution, architecture dispatch,
weather labels and offline copilot evaluation. One new fixture initially
omitted required copilot response fields; that fixture was corrected without
changing research values or the production copilot. The targeted ten tests
then passed. A final SVG validity/whitespace test was added after reviewing
the generated figures. The final full suite passed **111 tests**: 106 non-neural
tests in 23.46 s and 5 neural fixture tests in 16.94 s, in separate CPU processes.

Ruff lint, bytecode compilation, offline lock consistency, Git whitespace
checks and all four existing evidence-package checks passed. Formatting checks
pass for all eight touched/new Python files. The optional repository-wide
format check still reports **49 legacy files** requiring reformatting; these
pre-existing differences are not numerical/test failures. A blanket reformat
was deliberately avoided to preserve the scope and historical source hashes.
No configured static type checker is available. Existing pandas deprecation
warnings remain. No package upgrade was made.

Commands include `make test PY='.venv/bin/python'` with two CPU threads,
`.venv/bin/ruff check src app tests scripts experiments kaggle`,
`python -m compileall`, `uv lock --check --offline`, `git diff --check` and
`python -m gridcast.experiments.check_results` with `PYTHONPATH=src`.

## Completed versus pending research

The earlier no-weather benchmark, retrospective history/oracle controls,
timestamp availability safeguards, rolling/adaptive interval study and drift
diagnostics remain documented in their original reports. The adaptive crossing
and long-horizon undercoverage are negative results, not corrected away.
Authentic operational weather, fresh classical tuning, validation-only
interval selection and full neural/GPU benchmarking remain pending.

`NEXT_EXPERIMENTS.md` specifies questions, data/features, training and selection
rules, leakage risks, metrics, estimated compute and acceptance criteria for
five priorities. No new training, archive acquisition, GPU/cloud job, paid API
or infrastructure change was launched during this synchronization/export task.
Publication is a normal Git push only, with no deployment. Final branch tips,
commit IDs, remote containment and clean working-tree status are independently
checked and reported after the push.

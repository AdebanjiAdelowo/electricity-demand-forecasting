# Computational validation and outstanding work

## Executed experiments

* Five existing no-weather forecasting models, four expanding folds, horizons
  1/24/48/168 hours: `results/forecasting_baselines/20261012T093203Z/`.
  Raw-to-processed demand reconstruction passed; total run time 273.62 s.
* Five-bus Gaussian daily-window generation, raw/box/corrected comparison,
  three sampling seeds, independent AC check, utility evaluation and figures:
  `results/physics_informed_synthetic_energy/20261012T093721Z/`.
  Final run time 2.54 s; 67 pandapower checks passed.
* Prior physics run `20261012T093152Z` retained as earlier evidence. The final
  run changes the correction anchor to zero generation. Failed early
  forecasting runs remain on disk without completion metadata and are not
  included in results claims.

## Software checks

All 62 unique tests pass across separate CPU processes: the original
non-neural suite plus the nine synthetic-energy/preprocessing tests, and five
neural tests. Core checks use deterministic fixtures without network or GPU.
The added tests cover AC equations, losses, constraint/convergence distinction,
failure handling, correction, independent solver agreement, deterministic
generation, metric identity, causal utility features, chronological boundaries
and invariance of fitted Ridge/scaler parameters to post-training tampering.

Commands:

```sh
GRIDCAST_JOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLCONFIGDIR=results/scratch/matplotlib make test PY='.venv/bin/python'
.venv/bin/ruff check src app tests kaggle experiments
.venv/bin/python -m compileall -q src experiments tests
PYTHONPATH=src .venv/bin/python -m gridcast.experiments.check_results
uv lock --check --offline
git diff --check
```

Ruff, bytecode compilation, lock consistency, Git whitespace checks, and
consistency checks of all three completed evidence packages passed. Six
scientific figure types were visually inspected. No static type checker is
configured, so bytecode compilation must not be described as type checking.

The initial environment did not expose the editable package without
`PYTHONPATH=src`. Experiment commands explicitly set it and pytest now has
repository-specific Python paths. The optional pandapower dependency caused
a locked NumPy/pandas version change. An initial test attempt overlapping
that environment update crashed and is not counted as a completed test.
A subsequent combined suite stalled at LightGBM after running neural tests;
the stalled session-owned process was terminated. Separate fresh processes
passed all tests; Makefile and CI now use this separation. This observes a
native-runtime interaction but does not establish its exact root cause.
Existing pandas downcasting deprecation warnings remain.

## Git and preservation

The existing history and data are preserved. Only the new code, configuration,
tests, reports, README, CI and generated-output ignore rules belong to this
change. Pre-existing uncommitted `studies.py`, Kaggle runner, copilot evaluation,
plotting code, anomaly tests and historical results are left intact and must
not be included in this work's local commit. Generated profiles/checkpoints
and new prediction outputs remain outside Git. The environment remains about
1.2 GiB; each new experiment output family is approximately 16 MiB. No shared
cache cleanup, large download, GPU job, push or deployment occurred.

## Outstanding work, in priority order

1. Preserve the completed no-weather and small-network evidence for the
   application. CV and letter material is in the application summary.
2. Add renewable availability limits before interpreting generation as a
   physically complete photovoltaic model. Current network constraints do
   not enforce daylight availability; small nighttime generator artifacts
   remain a limitation.
3. Transfer the controlled comparison to a published feeder and richer
   independent reference realizations. Do not describe this custom feeder
   as a standard case or operational validation.
4. Verify archived-weather issue times and evaluate any neural forecasting
   run with a clearly matched information regime. Full neural/ERCOT training,
   stress, robustness and training-penalty studies remain unexecuted here.
5. Add independent training/process replications if stronger uncertainty
   claims are needed. Three sampling seeds do not capture these uncertainties.

These extensions are optional to the completed bounded proof of concept.
The reports make no claims of end-to-end physics-informed training, diffusion,
GNNs, differential privacy, real utility deployment or GPU execution.

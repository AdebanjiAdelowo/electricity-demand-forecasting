# Research summary for the Radboud application

## A. Project description

GridCast studies reproducible electricity-demand forecasting on hourly ERCOT
data and the feasibility of synthetic multivariate energy windows on a small
simulated electrical feeder. The completed forecasting comparison evaluates
five existing statistical and machine-learning baselines across four expanding
test folds, with leakage controls, probabilistic evaluation and paired temporal
bootstrap comparisons. A separate five-bus study tests a Gaussian window
generator, simple bound clipping and AC-feasibility correction on identical
samples. It measures statistical fidelity, network feasibility, diversity and
forecasting utility on independent held-out **simulated** days.

Evidence: `FORECASTING_BENCHMARK_REPORT.md` and
`PHYSICS_INFORMED_SYNTHETIC_ENERGY_REPORT.md`, with timestamped run metadata,
configuration files, predictions, numerical tables and scientific figures.

## B. Technical contributions

* Reconstructed cleaned demand from the raw EIA workbook and verified the
  processed demand values and imputation flags across 97,914 hourly records.
* Reused the existing forecasting models and fold infrastructure in a common
  no-weather benchmark, enforcing target-aware training and test boundaries,
  train-only scaling, fold-specific seasonal MASE and causal calibration.
* Added an independently checked vectorized balanced AC feeder solver and
  operating-limit checks, distinguishing equation mismatch, convergence and
  voltage/current/generation-bound feasibility.
* Implemented a compact Gaussian PCA generator and an explicit whole-window
  feasibility correction, with box-only ablation and three sampling seeds.
* Evaluated marginal/temporal/correlation/peak fidelity, diversity and matched
  downstream predictors, preserving negative findings and limitations.

These describe completed computational work. The forecasting baseline models
were already present; this contribution extends validation and evaluation
rather than claiming to have newly invented those architectures.

## C. Research relevance

The synthetic experiment addresses a central question in physics-aware data
generation: a statistically plausible sample need not represent an admissible
network operating point. Explicit AC equations and topology make that
distinction testable. Probabilistic generation supplies multivariate windows;
projection corrects network-limit violations without changing model training.
The ablation demonstrates why input bounds, power-flow consistency and
operating feasibility must be measured separately. Fidelity and downstream
utility reveal that improved feasibility alone is insufficient evidence of
better synthetic data.

The forecasting benchmark complements this through chronological evaluation
and empirical uncertainty assessment. Long-horizon interval undercoverage
illustrates the importance of checking calibration under changing demand.
These are relevant foundations for scientific machine learning and the SHARE
research direction described in the application brief; this summary does not
claim to reproduce SHARE's full setting or real utility datasets.

## D. Limitations

The network is an illustrative five-bus balanced feeder, not a named standard
benchmark or a deployed grid. The reference process conditions independent
simulated days on feasibility and excludes many stressed draws. The Gaussian
model is simple and can produce negative or small positive nighttime renewable
values; daylight availability is not enforced. Correction is post-processing,
not end-to-end physics-informed training. Three seeds vary sampling from one
fitted generator; they are not independent training replications.

Measured fully feasible days improve from 85.68% after box clipping to 100%
after network correction, across the three sampling seeds. Raw generator days
score 0% because small negative renewable values occur within each day; this
must not be used as an inflated claim of network-learning success. Corrected
samples lose some diversity and have one-hour downstream MAE 0.026380 MW,
compared with 0.024611 MW when the predictor trains on the original simulated
process. There is no validated real-operational utility, privacy result,
diffusion model, GNN, large-scale grid simulation, GPU benchmark or deployment.

## E. Three CV bullets

* Validated an ERCOT demand forecasting benchmark across four expanding test
  folds, achieving 2234 MW 24-hour MAE with Ridge, 8.6% below daily seasonal
  naive, using chronological evaluation and fold-specific preprocessing.
* Built a five-bus synthetic-energy experiment with a probabilistic window
  generator and AC feasibility correction; increased feasible simulated days
  from 85.7% after bound clipping to 100% across three sampling seeds.
* Evaluated synthetic energy data for statistical fidelity, temporal structure,
  diversity and held-out simulated forecasting utility; independently checked
  the AC solver against pandapower on 67 operating points.

## F. Motivation-letter paragraph

In my electricity-demand forecasting project, I developed a reproducible
evaluation of hourly ERCOT demand and extended it with a small experiment on
physics-aware synthetic energy data. I fitted a probabilistic generator of
multivariate daily windows, evaluated the generated profiles on a five-bus AC
network, and compared simple bound clipping with explicit feasibility
correction. The correction removed the observed network-limit violations,
but it also reduced diversity and did not improve the held-out simulated
forecasting task. This taught me to treat physical feasibility, statistical
realism and downstream usefulness as separate research objectives. I would
like to build on that foundation in the SHARE PhD, investigating generative
models that incorporate network structure and physical constraints while
preserving uncertainty and useful temporal variation.

# Physics-aware synthetic energy windows: a five-bus proof of concept

## Research question and motivation

Can post-generation correction improve the operating feasibility of a small
multivariate energy generator while retaining statistical and temporal fidelity?
The purpose is a controlled scientific experiment, not a solution to a full
utility data synthesis problem. We distinguish a learned probability model,
an AC simulator, and an operating-constraint correction.

## Simulator and data provenance

All network observations are simulated. None are Alliander, Dutch household,
or real operational records. The feeder is an **illustrative custom network**,
not MATPOWER case5 or a named benchmark. Its five buses permit an independently
checkable balanced AC formulation. The established independent solver is
[pandapower balanced AC power flow](https://pandapower.readthedocs.io/en/latest/powerflow/ac.html).

Three-phase bases: 1 MVA and 20 kV, hence impedance base 400 ohm and current
base 0.028868 kA. Bus 0 has fixed 1 p.u. complex slack voltage. Buses 1–4
have PQ loads; bus 4 also has unity-power-factor renewable generation.
There are no shunts, tap changers or PV-controlled buses.

| Line | Resistance p.u. | Reactance p.u. | Current limit p.u. |
|---|---:|---:|---:|
| 0–1 | 0.030 | 0.020 | 0.95 |
| 1–2 | 0.040 | 0.025 | 0.70 |
| 1–3 | 0.035 | 0.025 | 0.60 |
| 3–4 | 0.040 | 0.030 | 0.45 |

Each independent 24-hour day combines a shared lognormal demand multiplier,
bus-specific lognormal multipliers, a daily cosine profile, and AR(1) noise.
Reactive consumption is 0.25 times active consumption. Renewable production
has a daylight-shaped profile with a random daily amplitude and hourly noise.
Full coefficients are in `generator.py`. The reference distribution explicitly
conditions this process on **whole-day feasibility**, using rejection sampling:
128 training days accepted from 409 draws, and 64 independent held-out days
accepted from 163 draws. Different seeds define the two samples. This is a
stationary simulated-day split, not a chronology or a real-world season split.
The selection bias is part of the reference process and limits generalization.

## Mathematical formulation

Let (V_i) be complex voltage and (Y) the nodal admittance matrix. At
non-slack buses, consumption-positive specified power is

\[
S_i^{load}=P_i+j(0.25P_i)-G_i,
\qquad V_i\overline{(YV)_i}+S_i^{load}=0.
\]

The vectorized backward/forward sweep iterates branch currents from
\(\overline{S_i^{load}/V_i}\) and branch voltage drops \(Z_{ij}I_{ij}\).
We separately calculate the maximum complex nodal mismatch, in MVA, using
the admittance matrix. It is an equation residual, distinct from limit checks.
Convergence requires voltage-update error below (10^{-10}) p.u. and
nodal mismatch below (10^{-8}) MVA; 100 iterations is the maximum.

Operating limits are (0.95\le |V_i|\le1.05),

\[
|I_{ij}|\le I_{ij}^{max},\quad 0\le P_i\le2\text{ MW},
\quad0\le G_4\le1\text{ MW},
\quad |P_0|\le1.5\text{ MW},\quad |Q_0|\le0.75\text{ MVAr}.
\]

Thermal limits are current limits, not apparent-power caps. Solver convergence
alone cannot establish operating feasibility. A trajectory passes only if
every hourly state converges and passes every stated operating constraint.

## Generative model definitions

Baseline A fits one Gaussian PCA model to flattened 24×5 windows, standardized
using **training** means and standard deviations (minimum scale 0.02 MW).
There are 12 retained principal components and orthogonal Gaussian residual
noise estimated from discarded variance. With training mean \(\mu\), diagonal
scale \(D\), components \(W\), factor variances \(\Lambda\), and residual variance
\(\sigma^2\), the sampler is

\[
X=\mu+D\{W^T\Lambda^{1/2}z+\sigma(I-W^TW)\epsilon\},
\quad z,\epsilon\sim\mathcal N(0,I).
\]

The PCA objective minimizes standardized squared reconstruction error for an
orthonormal rank-12 subspace. The model is probabilistic and linear, not a VAE,
diffusion model or graph neural network. It has no physical training penalty.

The box-only ablation clips consumption and renewable production to their
bounds. Baseline B applies that clipping, then contracts each complete window
toward a verified feasible anchor \(A=(0.1,0.1,0.1,0.1,0)\) MW:

\[
\widetilde X=A+\alpha(X_{box}-A),\qquad0\le\alpha\le1.
\]

Eighteen bisection steps maintain an explicitly verified feasible endpoint.
One coefficient is used per window to preserve temporal shape. The algorithm
does not claim a globally closest-point projection or globally monotone AC
feasibility. It always verifies the final returned samples. This correction
changes the sampled distribution **after fitting**; it is not end-to-end
physics-informed training. Voltage and line states in every arm are derived
from the simulator, so all converged arms already satisfy its AC equations.

## Experimental protocol and reproducibility

Executed CPU run: `results/physics_informed_synthetic_energy/20261012T093721Z/`.
The run records HEAD `b1c92e37cbea546ca6bdd0eb8932f2c5cfd93795`, dirty status,
source-file SHA-256 hashes, lock hash, reference-file fingerprint, package
versions, configuration, seeds and timing. Dirty-run source hashes identify
the executed code; HEAD alone is not a complete description of that code.

Use the existing project-specific environment and uv lockfile:

```sh
uv sync --locked --extra physics --extra dev
PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MPLCONFIGDIR=results/scratch/matplotlib \
  .venv/bin/python -m experiments.physics_informed_synthetic_energy.run
```

`config.json` defines 128 generated days per arm and sampling seeds 0, 1, 2.
All arms share one deterministic trained model and the same latent samples
within each seed. Seed variation measures sampling variation, not repeated
training or parameter uncertainty. No GPU execution occurred.
PCA fit took 0.1370 s; each projection batch took 0.0434–0.0474 s; total
runner time including solver validation and figure generation was 2.54 s.
These are observed wall times on this machine, not portable speed guarantees.

## Quantitative results and ablation

Means over the three sampling seeds; reference-trained rows use the same
training sample repeatedly and are one measurement, not three replications.

| Measure | Unconstrained | Box only | Feasibility correction |
|---|---:|---:|---:|
| Fully feasible days (%) | 0.00 | 85.68 | 100.00 |
| Feasible hours (%) | 71.08 | 98.58 | 100.00 |
| Convergence (%) | 100.00 | 100.00 | 100.00 |
| Voltage-limit violations, hours (%) | 0.901 | 0.803 | 0.000 |
| Line-overload hours (%) | 0.629 | 0.629 | 0.000 |
| Input-bound violation hours (%) | 27.919 | 0.000 | 0.000 |
| Slack-limit violation hours (%) | 0.000 | 0.000 | 0.000 |
| Marginal Wasserstein distance (MW) | 0.008239 | 0.007874 | 0.007813 |
| Mean absolute mean difference (MW) | 0.006944 | 0.007299 | 0.007198 |
| Mean absolute variance difference (MW²) | 0.001062 | 0.000982 | 0.000787 |
| ACF absolute difference | 0.043626 | 0.043594 | 0.043594 |
| Cross-correlation absolute difference | 0.039508 | 0.039502 | 0.041401 |
| Daily-peak Wasserstein distance (MW) | 0.021030 | 0.021030 | 0.019649 |
| Pairwise RMS diversity (MW) | 0.057104 | 0.056930 | 0.054720 |
| Mean absolute hourly change (MW) | 0.021560 | 0.021036 | 0.020671 |
| Near-duplicate pair fraction (<0.001 MW RMS) | 0 | 0 | 0 |

Complete metrics and per-seed ranges are in `metrics.csv` and `summary.csv`.
Box-only day feasibility ranges from 85.16% to 85.94%. Corrected marginal
distance ranges from 0.007228 to 0.008153 MW. The main evidence is improvement
over **box-only**, because the zero raw whole-day score is dominated by
small negative renewable values generated near nighttime zero production.
Physical correction improves full feasibility and slightly improves marginal
fidelity, while reducing diversity and worsening cross-correlation fidelity
relative to box-only. Improvements are metric-specific,
not universal. Identical box and corrected ACF is expected because a positive
affine transformation of each trajectory preserves its centred ACF.

All observed maximum nodal residuals are below (2.5\times10^{-11}) MVA.
The independent pandapower Newton–Raphson check covered 67 systematic and
extreme operating points with zero failures, maximum voltage difference

\[
1.095\times10^{-11}\text{ p.u.},\qquad
\max |\Delta\text{line loading}|=1.974\times10^{-10}.
\]

This is strong numerical agreement on sampled states, not exhaustive solver
validation on every possible operating regime. All reported states are saved
with complex voltages, line currents/loading, slack powers, residuals,
convergence and violation flags.

## Downstream utility

An identical fixed-alpha Ridge predictor uses the preceding three aggregate
load values and known sine/cosine hour features to predict the next hour.
Training is on 128 days from each arm; evaluation is on the same 64 independent
held-out simulated days (1,344 forecast targets). Scalers fit only on the
chosen training source. Forecast pairs never cross day boundaries.

| Forecast training source | MAE (MW) | RMSE (MW) |
|---|---:|---:|
| Original simulated training process | 0.024611 | 0.030762 |
| Unconstrained generated windows | 0.026297 | 0.033215 |
| Box-only generated windows | 0.026298 | 0.033216 |
| Physically corrected windows | 0.026380 | 0.033310 |

Correction did not improve this forecasting proxy. Its MAE is approximately
7.2% above training on the original simulated process. This is
train-synthetic/test-held-out-simulation evaluation; it is not test-real
operational utility. Three sampling seeds do not establish statistical
significance, and all share one held-out sample.

## Figures

PNG and editable SVG plots are in the executed output directory. Captions
are saved in `captions.json`.

* `load_profiles`: mean total load and 10–90% day envelopes, sampling seed 0.
* `voltage_distributions`: non-slack voltage distributions and operating limits.
* `autocorrelation`: within-day, daily-mean-centred ACF across five variables.
* `statistical_fidelity`: per-seed marginal, ACF and peak distances.
* `feasibility_fidelity`: hourly violations and day feasibility/fidelity tradeoff.
* `downstream_forecasting`: held-out simulated MAE by training source.

## Failure cases and limitations

The Gaussian model gives negative renewable values near zero, making the raw
whole-day feasibility metric especially unforgiving. Box clipping explains
most of the initial gain. Gaussian samples and box correction may retain small
positive nighttime renewable values; there is no daylight availability constraint.
The final anchor has zero generation, so contraction does not introduce
additional nighttime output. Remaining nighttime output is a statistical realism
failure, even though it passes the stated network constraints. The study
does not assert physical completeness for a specific renewable technology.

The feeder is balanced, small, static and illustrative, with no stochastic
topology, phase imbalance, dynamic response, frequency model or network
contingencies. Bounds and reference process are modelling assumptions.
Reference rejection removes stressed days. The generator sees only 128 days
and captures a simple stationary process. No seasonality beyond the daily
shape, demand growth, parameter uncertainty, privacy guarantee or real meter
distribution has been established. Duplicate thresholds are diagnostics,
not proof against mode collapse. The nearest feasible solution is not solved.
Only 67 states receive independent solver verification. No training-penalty
ablation was executed; the executed ablation is raw → box-only → correction.

## Future work

Add resource-aware renewable bounds, then
test on a published feeder and a richer independent reference process. Evaluate
reference-process uncertainty over independent train/test replications, not
only sampling seeds. A later controlled conditional neural generator could
compare a genuine differentiable penalty against the same post-processing
baseline, with actual-solver checks and matched budgets. Each extension
should preserve the feasibility, fidelity, diversity and utility distinction.

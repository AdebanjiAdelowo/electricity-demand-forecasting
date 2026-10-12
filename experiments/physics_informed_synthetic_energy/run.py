"""Run with python -m experiments.physics_informed_synthetic_energy.run."""
from __future__ import annotations

import argparse
import json
import time
from importlib import metadata
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from gridcast import config, tracking
from gridcast.experiments.revalidate import fingerprint

from .evaluation import acf, evaluate, utility
from .generator import GaussianWindows, reference
from .network import ANCHOR, PARENTS, Q_RATIO, RATINGS, Z, pandapower_check, project, solve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def figures(out: Path, datasets: dict, scores: pd.DataFrame) -> dict:
    captions = {}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    def save(name, fig, caption):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=180)
        fig.savefig(out / f"{name}.svg")
        plt.close(fig)
        captions[name] = caption

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, a in datasets.items():
        totals = a[:, :, :4].sum(axis=-1)
        ax.plot(np.arange(24), totals.mean(axis=0), label=label)
        ax.fill_between(np.arange(24), *np.quantile(totals, [0.1, 0.9], axis=0), alpha=0.12)
    ax.set(xlabel="Hour of simulated day", ylabel="Total consumption (MW)")
    ax.legend()
    save("load_profiles", fig, "Mean and 10–90% envelopes across independent simulated days; sampling seed 0.")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, a in datasets.items():
        ax.hist(abs(solve(a)["voltage"])[..., 1:].ravel(), bins=np.linspace(0.9, 1.08, 60),
                density=True, histtype="step", label=label)
    ax.axvline(0.95, color="black", ls=":")
    ax.axvline(1.05, color="black", ls=":")
    ax.set(xlabel="Non-slack voltage magnitude (p.u.)", ylabel="Probability density (1/p.u.)")
    ax.legend()
    save("voltage_distributions", fig, "Non-slack voltages and operating limits, seed 0; converged states.")
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, a in datasets.items():
        ax.plot(np.arange(1, 13), acf(a).mean(axis=1), marker=".", label=label)
    ax.set(xlabel="Within-day lag (hours)", ylabel="Mean autocorrelation across five variables")
    ax.legend()
    save("autocorrelation", fig, "Daily-mean-centred ACF, averaged across trajectories and variables; no cross-day pairs.")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, metric, ylabel in zip(axes, ["marginal_wasserstein_mw", "acf_absolute_difference", "peak_wasserstein_mw"],
                                  ["Marginal W1 (MW)", "Absolute ACF difference", "Daily peak W1 (MW)"], strict=True):
        for i, method in enumerate(["unconstrained", "box_only", "projected"]):
            values = scores[scores.method == method][metric].to_numpy()
            ax.scatter(np.full(len(values), i), values, alpha=0.7)
            ax.plot(i, values.mean(), marker="_", color="black", markersize=16)
        ax.set(xticks=[0, 1, 2], xticklabels=["Raw", "Box", "Projected"], ylabel=ylabel)
    save("statistical_fidelity", fig, "Points are three sampling seeds; bars mark means. Lower distances indicate closer held-out fidelity.")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    means = scores.groupby("method").mean(numeric_only=True).reindex(["unconstrained", "box_only", "projected"])
    for metric in ["voltage_violation_rate", "line_overload_rate", "bounds_violation_rate"]:
        axes[0].plot(np.arange(3), 100 * means[metric], marker="o", label=metric.replace("_violation_rate", "").replace("_rate", ""))
    axes[0].set(xticks=np.arange(3), xticklabels=["Raw", "Box", "Projected"], ylabel="Violating hours (%)")
    axes[0].legend(fontsize=8)
    for method, g in scores[scores.method != "reference_heldout"].groupby("method"):
        axes[1].scatter(g.marginal_wasserstein_mw, 100 * g.trajectory_feasible_fraction, label=method)
    axes[1].set(xlabel="Marginal W1 to held-out simulation (MW)", ylabel="Fully feasible days (%)")
    axes[1].legend(fontsize=8)
    save("feasibility_fidelity", fig, "Hourly violations averaged across seeds and whole-day feasibility versus marginal fidelity.")
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(["reference_train", "unconstrained", "box_only", "projected"]):
        vals = scores[scores.method == method].utility_mae_mw
        ax.scatter(np.full(len(vals), i), vals, label=method)
    ax.set(xticks=np.arange(4), xticklabels=["Reference", "Raw", "Box", "Projected"],
           ylabel="Held-out simulated one-hour MAE (MW)", xlabel="Forecasting training source")
    save("downstream_forecasting", fig, "Same Ridge predictor, training sample size and held-out simulated days; this is a utility proxy.")
    (out / "captions.json").write_text(json.dumps(captions, indent=2))
    return captions


def run(settings: dict) -> Path:
    t0 = time.perf_counter()
    out = config.RESULTS / "physics_informed_synthetic_energy" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(settings, indent=2))
    train, train_meta = reference(settings["train_trajectories"], settings["train_seed"], settings["hours"])
    heldout, test_meta = reference(settings["heldout_trajectories"], settings["heldout_seed"], settings["hours"])
    fit_start = time.perf_counter()
    model = GaussianWindows().fit(train, settings["pca_rank"])
    fit_seconds = time.perf_counter() - fit_start
    np.savez_compressed(out / "reference.npz", train=train, heldout=heldout)
    np.savez_compressed(out / "model.npz", mean=model.mean, scale=model.scale, components=model.pca.components_,
                        factor_variance=model.pca.explained_variance_, residual_variance=model.noise)
    rows, checks, sampling_times, projection_times = [], [], [], []
    for seed in settings["sampling_seeds"]:
        print(f"Physics study seed {seed}", flush=True)
        sample_start = time.perf_counter()
        raw = model.sample(settings["generated_trajectories"], seed)
        sampling_times.append(time.perf_counter() - sample_start)
        projection_start = time.perf_counter()
        corrected, alpha = project(raw, settings["projection_steps"])
        projection_times.append(time.perf_counter() - projection_start)
        box = np.clip(raw, 0, [2, 2, 2, 2, 1])
        for label, a in {"unconstrained": raw, "box_only": box, "projected": corrected,
                         "reference_train": train, "reference_heldout": heldout}.items():
            row = {"seed": seed, "method": label, **evaluate(a, heldout), **utility(a, heldout)}
            # Heldout-vs-itself fidelity and heldout-trained utility are not valid independent estimates.
            if label == "reference_heldout":
                for key in list(row):
                    if key.startswith("utility_") or "difference" in key or "wasserstein" in key:
                        row[key] = np.nan
            rows.append(row)
            state = solve(a)
            np.savez_compressed(out / f"{label}_seed{seed}.npz", profiles=a, **state)
        np.save(out / f"projection_alpha_seed{seed}.npy", alpha)
        checks += [raw.reshape(-1, 5), corrected.reshape(-1, 5)]
        if seed == settings["sampling_seeds"][0]:
            plot_data = {"Held-out reference": heldout, "Unconstrained": raw, "Projected": corrected}
    check_points = np.concatenate([train.reshape(-1, 5), heldout.reshape(-1, 5), *checks])
    selected = np.linspace(0, len(check_points) - 1, settings["independent_solver_points"], dtype=int)
    # Include extreme load and renewable states in addition to the systematic sample.
    extremes = [np.argmax(check_points[:, :4].sum(axis=1)), np.argmin(check_points[:, 4]), np.argmax(check_points[:, 4])]
    checked = pandapower_check(check_points[np.unique(np.r_[selected, extremes])])
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "metrics.csv", index=False)
    scores.groupby("method").agg(["mean", "std", "min", "max"]).to_csv(out / "summary.csv")
    figures(out, plot_data, scores)
    source = Path(__file__).parent
    record = {"git": tracking.git_state(), "device": "cpu", "config": settings,
              "versions": {p: metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "pandapower", "matplotlib"]},
              "source_sha256": {str(p.relative_to(config.ROOT)): fingerprint(p) for p in source.glob("*.py")},
              "lock_sha256": fingerprint(config.ROOT / "uv.lock"),
              "reference_sha256": fingerprint(out / "reference.npz"),
              "train_process": train_meta, "heldout_process": test_meta,
              "fit_seconds": fit_seconds, "sampling_seconds": sampling_times,
              "projection_seconds": projection_times, "independent_solver_check": checked,
              "wall_seconds": time.perf_counter() - t0,
              "variation": "Three sampling seeds, one deterministic training fit; not independent training replications",
              "topology": {"parent_buses": PARENTS.tolist(), "child_buses": [1, 2, 3, 4],
                           "z_pu": [[z.real, z.imag] for z in Z], "current_ratings_pu": RATINGS.tolist(),
                           "base_mva": 1, "base_kv": 20, "slack_voltage_pu": 1,
                           "load_q_p_ratio": Q_RATIO, "anchor_mw": ANCHOR.tolist()}}
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    print(scores[["seed", "method", "trajectory_feasible_fraction", "marginal_wasserstein_mw", "utility_mae_mw"]].to_string(index=False))
    print(out, flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    run(json.loads(parser.parse_args().config.read_text()))

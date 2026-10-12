"""Static figures for the documentation, drawn only from the stored result tables.

Conventions: one y-axis per panel, thin marks, recessive grid, colour encodes
identity in a fixed order, and series are labelled directly where there are
few enough to do so. Every figure has a CSV behind it in results/tables.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gridcast import config  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FAMILY = {"statistical": SERIES[0], "ml": SERIES[1], "deep": SERIES[2], "foundation": SERIES[6]}
FAMILY_LABEL = {
    "statistical": "Statistical",
    "ml": "Machine learning",
    "deep": "Deep learning",
    "foundation": "Foundation",
}
OPERATIONAL = ["none"]  # legacy lead buckets do not establish forecast publication availability


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "legend.frameon": False,
            "lines.linewidth": 2,
            "figure.dpi": 120,
        }
    )


def _save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", f"results/figures/{name}.png")


def _t(name: str) -> pd.DataFrame | None:
    p = config.TABLES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _op(t: pd.DataFrame) -> pd.DataFrame:
    return t[t["regime"].isin(OPERATIONAL)]


def headline_bars() -> None:
    t = _t("benchmark_overall")
    if t is None:
        return
    t = _op(t)
    ref = _t("benchmark_vs_ercot_day_ahead")
    fig, axes = plt.subplots(1, 4, figsize=(15, 5.2), sharey=False)
    for ax, h in zip(axes, config.HEADLINE_HORIZONS, strict=True):
        g = t[t["h"] == h].sort_values("mae", ascending=False)
        ax.barh(g["model"], g["mae"], color=[FAMILY.get(f, MUTED) for f in g["family"]], height=0.62)
        for y, v in enumerate(g["mae"]):
            ax.text(v, y, f" {v:,.0f}", va="center", fontsize=8, color=MUTED)
        ax.set_title(f"{h} h ahead")
        ax.set_xlabel("MAE (MW)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, g["mae"].max() * 1.18)
        if h == 24 and ref is not None and (ref["model"] == "ercot_day_ahead_reference").any():
            v = float(ref.loc[ref["model"] == "ercot_day_ahead_reference", "mae"].iloc[0])
            ax.axvline(v, color=INK, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(
                v,
                len(g) - 0.4,
                f" ERCOT published\n day-ahead forecast\n {v:,.0f} MW",
                fontsize=8,
                va="top",
                color=INK,
            )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in FAMILY.items() if f in set(t["family"])]
    fig.legend(
        handles,
        [FAMILY_LABEL[f] for f in FAMILY if f in set(t["family"])],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Test MAE by model and horizon, operational information only (July 2024 to June 2026)",
        x=0.01,
        y=1.07,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "headline_mae")


def horizon_profile() -> None:
    t = _t("benchmark_horizon_profile")
    if t is None:
        return
    t = _op(t)
    want = [
        "seasonal_naive_24h",
        "sarimax",
        "ridge",
        "lightgbm",
        "dshw",
        "sarimax_weather",
        "ridge_weather",
        "lightgbm_weather",
        "nbeats_weather",
        "tcn_weather",
        "gru_weather",
    ]
    names = [m for m in want if m in set(t["model"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(names):
        g = t[t["model"] == m].sort_values("h")
        ax.plot(g["h"], g["mae"], color=SERIES[i], marker="o", markersize=4, label=m)
        ax.text(g["h"].iloc[-1] + 2, g["mae"].iloc[-1], m, color=INK, fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_xticks(config.DENSE_HORIZONS)
    ax.set_xticklabels([str(h) for h in config.DENSE_HORIZONS], fontsize=8)
    ax.minorticks_off()
    ax.set_xlim(0.9, 330)
    ax.set_ylim(0, None)
    ax.set_xlabel("Forecast horizon (hours, log scale)")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Error growth with horizon, operational information only")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    _save(fig, "horizon_profile")


def weather_regimes() -> None:
    t = _t("benchmark_overall")
    if t is None:
        return
    pairs = [
        ("lightgbm", "lightgbm_weather", "LightGBM"),
        ("nbeats", "nbeats_weather", "N-BEATS-style"),
        ("ridge", "ridge_weather", "Ridge"),
    ]
    pairs = [p for p in pairs if {p[0], p[1]} <= set(t["model"])]
    if not pairs:
        return
    regimes = [
        ("none", "No weather", MUTED),
        ("forecast_raw", "Raw NWP (availability unverified)", SERIES[3]),
        ("forecast", "Corrected NWP (availability unverified)", SERIES[0]),
        ("oracle", "Reanalysis oracle, not operational", SERIES[1]),
    ]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.4), sharey=True, squeeze=False)
    for ax, (base, wx, label) in zip(axes[0], pairs, strict=True):
        x = np.arange(len(config.HEADLINE_HORIZONS))
        rows = []
        for r, lab, col in regimes:
            g = t[(t["model"] == (base if r == "none" else wx)) & (t["regime"] == r)].set_index("h")["mae"]
            if len(g):
                rows.append((lab, col, [g.get(h, np.nan) for h in config.HEADLINE_HORIZONS]))
        w = 0.8 / len(rows)
        for i, (lab, col, vals) in enumerate(rows):
            ax.bar(x + (i - (len(rows) - 1) / 2) * w, vals, w * 0.9, color=col, label=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h} h" for h in config.HEADLINE_HORIZONS])
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Horizon")
    axes[0][0].set_ylabel("MAE (MW)")
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle(
        "Value of weather information by how it is obtained", x=0.01, y=1.1, ha="left", fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "weather_regimes")


def calibration() -> None:
    t = _t("probabilistic_overall")
    if t is None:
        return
    model = "lightgbm_weather"
    g = t[t["model"] == model]
    methods = [
        m
        for m in ["raw_quantile", "cqr_static", "cqr_rolling", "conformal_static", "conformal_rolling"]
        if m in set(g["method"])
    ]
    nominal = [50, 80, 90, 95]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, h in zip(axes, config.HEADLINE_HORIZONS, strict=True):
        ax.plot([45, 100], [45, 100], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        for i, m in enumerate(methods):
            r = g[(g["h"] == h) & (g["method"] == m)]
            if r.empty:
                continue
            ax.plot(
                nominal,
                [100 * float(r[f"cov{n}"].iloc[0]) for n in nominal],
                color=SERIES[i],
                marker="o",
                markersize=5,
                label=m.replace("_", " "),
            )
        ax.set_title(f"{h} h ahead")
        ax.set_xlabel("Nominal coverage (%)")
        ax.set_xticks(nominal)
        ax.set_xlim(45, 100)
        ax.set_ylim(35, 100)
    axes[0].set_ylabel("Empirical coverage on test (%)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle(
        f"Prediction-interval calibration of {model} (dashed line = perfectly calibrated)",
        x=0.01,
        y=1.13,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "calibration")


def compute_tradeoff() -> None:
    t = _t("benchmark_compute")
    if t is None:
        return
    t = t[t["fit_seconds_per_fold"] > 0.5]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for fam, g in t.groupby("family"):
        ax.scatter(
            g["fit_seconds_per_fold"],
            g["mean_mae_headline"],
            s=70,
            color=FAMILY.get(fam, MUTED),
            edgecolor=SURFACE,
            linewidth=2,
            label=FAMILY_LABEL.get(fam, fam),
            zorder=3,
        )
    for r in t.itertuples():
        ax.annotate(
            r.model,
            (r.fit_seconds_per_fold, r.mean_mae_headline),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    front = t[t["pareto_optimal"]].sort_values("fit_seconds_per_fold")
    ax.plot(
        front["fit_seconds_per_fold"],
        front["mean_mae_headline"],
        color=MUTED,
        linewidth=1,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Training time per fold (seconds, log scale; hardware differs by family, see table)")
    ax.set_ylabel("Mean test MAE over 1, 24, 48, 168 h (MW)")
    ax.set_title("Accuracy versus training cost (dashed: Pareto front)")
    ax.legend(loc="upper right")
    _save(fig, "compute_tradeoff")


def anomaly_sensitivity() -> None:
    t = _t("anomaly_simulated")
    if t is None:
        return
    kinds = [k for k in ["spike", "dip_block", "level_shift"] if k in set(t["type"])]
    if not kinds:
        return
    dets = [
        d
        for d in [
            "robust_seasonal",
            "forecast_residual_h1",
            "forecast_residual_h24_weather",
            "cusum_h24_weather",
            "isolation_forest",
        ]
        if d in set(t["detector"])
    ]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4.2), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], kinds, strict=True):
        for i, d in enumerate(dets):
            g = t[(t["type"] == k) & (t["detector"] == d)].sort_values("magnitude")
            ax.plot(
                100 * g["magnitude"],
                100 * g["event_recall"],
                color=SERIES[i],
                marker="o",
                markersize=5,
                label=d.replace("_", " "),
            )
        ax.set_title(k.replace("_", " "))
        ax.set_xlabel("Injected size (% of demand)")
        ax.set_xticks([2, 5, 10, 20])
        ax.set_ylim(-3, 103)
    axes[0][0].set_ylabel("Simulated events detected (%)")
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06), fontsize=9)
    fig.suptitle(
        "Detector sensitivity on SIMULATED anomalies at a 0.5% clean-hour alarm rate",
        x=0.01,
        y=1.13,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "anomaly_sensitivity")


def example_week(
    start: str = "2024-07-05", days: int = 9, model: str = "lightgbm_weather", h: int = 24
) -> None:
    p = config.RESULTS / "serving" / "forecasts.parquet"
    if not p.exists():
        return
    f = pd.read_parquet(p)
    a = pd.Timestamp(start, tz="UTC")
    g = f[
        (f["model"] == model)
        & (f["h"] == h)
        & (f["target_time"] >= a)
        & (f["target_time"] < a + pd.Timedelta(days=days))
    ]
    if g.empty:
        return
    x = g["target_time"].dt.tz_convert(config.LOCAL_TZ).dt.tz_localize(None)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(
        x, g["q0.05"], g["q0.95"], color=SERIES[0], alpha=0.18, linewidth=0, label="90% prediction interval"
    )
    ax.plot(x, g["yhat"], color=SERIES[0], label=f"{model}, {h} h ahead")
    ax.plot(x, g["y"], color=INK, linewidth=1.5, label="Actual demand")
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Local time (America/Chicago)")
    ax.set_title("Stored day-ahead forecasts: retrospective legacy weather inputs")
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    _save(fig, "example_week_beryl")


def main(argv=None) -> None:
    _style()
    for fn in (
        headline_bars,
        horizon_profile,
        weather_regimes,
        calibration,
        compute_tradeoff,
        anomaly_sensitivity,
        example_week,
    ):
        fn()


if __name__ == "__main__":
    main()

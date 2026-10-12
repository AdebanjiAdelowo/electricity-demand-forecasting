"""Short CPU Ridge runs plus post-hoc analysis of immutable validated forecasts."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from gridcast import config, tracking
from gridcast.data import dataset
from gridcast.evaluation import backtest, metrics
from gridcast.evaluation.shift import residual_drift, slices
from gridcast.experiments.revalidate import fingerprint, paired_daily_bootstrap
from gridcast.features.available_weather import WeatherConfig
from gridcast.models.available_weather import WeatherRidge
from gridcast.probabilistic.conformal import calibrate
from gridcast.probabilistic.online import OnlineConfig, calibrate_online

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def static_intervals(pred: pd.DataFrame) -> pd.DataFrame:
    wide = calibrate(pred, "abs", "static")
    rows = []
    for alpha in config.INTERVAL_ALPHAS:
        p = wide.copy()
        p["alpha"], p["lo"], p["hi"] = alpha, p[f"q{alpha / 2:g}"], p[f"q{1 - alpha / 2:g}"]
        p["method"] = "conformal_static"
        rows.append(p.drop(columns=metrics.qcols()))
    return pd.concat(rows, ignore_index=True)


def interval_table(intervals: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in intervals.groupby([*by, "alpha"], observed=True):
        y, lo, hi = g.y.to_numpy(), g.lo.to_numpy(), g.hi.to_numpy()
        alpha = key[-1]
        rows.append({**dict(zip([*by, "alpha"], key, strict=True)), "n": len(g),
                     "nominal": 1 - alpha, "coverage": float(np.mean((y >= lo) & (y <= hi))),
                     "width_mw": float(np.mean(hi - lo)),
                     "winkler_mw": float(np.mean(metrics.interval_score(y, lo, hi, alpha))),
                     "coverage_error": float(np.mean((y >= lo) & (y <= hi)) - (1 - alpha))})
    return pd.DataFrame(rows)


def interval_pinball(intervals: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "regime", "fold", "h", "origin", "method"]
    rows = []
    for (model, regime, h, method), g in intervals.groupby(["model", "regime", "h", "method"], observed=True):
        reference = g[g.alpha == 0.1].set_index(keys)
        cols = {0.5: reference.yhat}
        for alpha in config.INTERVAL_ALPHAS:
            p = g[g.alpha == alpha].set_index(keys).reindex(reference.index)
            cols[alpha / 2] = p.lo
            cols[1 - alpha / 2] = p.hi
        q = np.column_stack([cols[level].to_numpy() for level in config.QUANTILES])
        valid = np.isfinite(q).all(axis=1)
        rows.append({"model": model, "regime": regime, "h": h, "method": method, "n": int(valid.sum()),
                     "pinball_mw": metrics.pinball(reference.y.to_numpy()[valid], q[valid], config.QUANTILES),
                     "quantile_crossing_fraction": float((np.diff(q[valid], axis=1) < 0).any(axis=1).mean())})
    return pd.DataFrame(rows)


def paired_weather(scored: pd.DataFrame, baseline: str = "ridge", seed: int = 0) -> pd.DataFrame:
    """Reuse the seven-day within-fold bootstrap on exact matched prediction keys."""
    key = ["fold", "origin", "h"]
    base = scored[scored.model == baseline].set_index(key)
    rows = []
    for name in ["ridge_history_proxy", "ridge_oracle"]:
        g = scored[scored.model == name].set_index(key)
        if len(g) != len(base) or not g.index.sort_values().equals(base.index.sort_values()):
            raise ValueError("Weather/model forecast keys differ")
        combined = pd.concat([base.reset_index().assign(model="seasonal_naive_week"), g.reset_index()], ignore_index=True)
        result = paired_daily_bootstrap(combined, seed=seed)
        result = result[result.model == name].rename(columns={"mae_difference_vs_weekly_mw": "mae_difference_vs_ridge_mw"})
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def plots(out: Path, point: pd.DataFrame, conditional: pd.DataFrame, intervals: pd.DataFrame):
    captions = {}

    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=180)
        fig.savefig(out / f"{name}.svg")
        plt.close(fig)
        captions[name] = caption

    fig, ax = plt.subplots(figsize=(8, 4))
    for model, g in point.groupby("model"):
        ax.plot(g.h, g.mae, marker="o", label=model)
    ax.set(xlabel="Forecast horizon (hours)", ylabel="MAE (MW)", xscale="log")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "horizon_error", "Matched forecast keys, four expanding folds; history_proxy and oracle are retrospective.")
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, g in conditional[conditional.h == 24].groupby("model"):
        ax.plot(g.season, g.mae, marker="o", label=model)
    ax.set(xlabel="Target season (Chicago interval start)", ylabel="24-hour MAE (MW)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "seasonal_error", "24-hour seasonal errors; causal demand features and retrospectively labelled weather controls.")
    for field, ylabel, name in [("coverage", "Empirical coverage (%)", "calibration_coverage"),
                                ("width_mw", "Mean 90% interval width (MW)", "interval_width")]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, model in zip(axes, ["ridge", "lightgbm"], strict=True):
            for method, g in intervals[(intervals.model == model) & (intervals.alpha == 0.1)].groupby("method"):
                ax.plot(g.h, g[field] * (100 if field == "coverage" else 1), marker="o", label=method)
            if field == "coverage":
                ax.axhline(90, ls=":", color="black", label="nominal 90%")
            ax.set(title=model, xlabel="Horizon (hours)", ylabel=ylabel, xscale="log")
            ax.legend(fontsize=7)
        save(fig, name, "Unchanged validated point forecasts; static versus timestamp-aware rolling and bounded delayed ACI.")
    (out / "captions.json").write_text(json.dumps(captions, indent=2))


def run(settings: dict, train_weather: bool = True) -> Path:
    start = time.perf_counter()
    out = config.RESULTS / "weather_uncertainty" / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps({**settings, "train_weather": train_weather}, indent=2))
    baseline_dir = config.ROOT / settings["baseline_directory"]
    old_record = json.loads((baseline_dir / "run.json").read_text())
    if fingerprint(dataset.PROCESSED) != old_record["data"]["processed_sha256"]:
        raise ValueError("Data changed since validated baseline")
    original_files = sorted(baseline_dir.glob("*.parquet"))
    original_hashes = {str(p.relative_to(config.ROOT)): fingerprint(p) for p in original_files}
    df = dataset.load()
    frames = [pd.read_parquet(p) for p in original_files]
    times, artifacts = {}, []
    if train_weather:
        for mode in settings["weather_modes"]:
            cfg = WeatherConfig(mode=mode, variables=tuple(settings["weather_variables"]),
                                observation_delay_hours=settings["observation_delay_hours"],
                                lags=tuple(settings["weather_lags"]), rolling_hours=tuple(settings["weather_rolling_hours"]),
                                fill_limit_hours=settings["fill_limit_hours"])
            if mode not in {"history_proxy", "oracle"}:
                raise ValueError("Real-data weather run requires history proxy or explicitly retrospective oracle")
            models = []

            def factory(c=cfg, fitted=models):
                model = WeatherRidge(c)
                fitted.append(model)
                return model

            print(f"Short CPU fit: Ridge {mode}", flush=True)
            pred, timing = backtest.run_backtest(factory, df, config.TEST_FOLDS, config.HEADLINE_HORIZONS)
            pred.to_parquet(out / f"ridge_{mode}.parquet", index=False)
            for model, fold in zip(models, config.TEST_FOLDS, strict=True):
                for h, pipeline in model.models.items():
                    path = out / f"ridge_{mode}_{fold.name}_h{h}.npz"
                    np.savez_compressed(path, scaler_mean=pipeline.named_steps["standardscaler"].mean_,
                                        scaler_scale=pipeline.named_steps["standardscaler"].scale_,
                                        coef=pipeline.named_steps["ridge"].coef_, intercept=pipeline.named_steps["ridge"].intercept_,
                                        features=np.array(model.feature_names[h]), medians=model.medians[h].to_numpy())
                    artifacts.append(path.name)
            frames.append(pred)
            times[mode] = timing
    pred = pd.concat(frames, ignore_index=True)
    scored = backtest.scoreable(pred)
    ends = {f.name: f.test_end for f in config.TEST_FOLDS}
    scored = scored[scored.target_time < scored.fold.map(ends)].copy()
    keys = ["fold", "origin", "h"]
    base_keys = scored[scored.model == "ridge"].set_index(keys).index.sort_values()
    for name, g in scored.groupby("model"):
        if not g.set_index(keys).index.sort_values().equals(base_keys):
            raise ValueError(f"Unmatched model keys: {name}")
    scales = {f.name: metrics.naive_scale(df.loc[(df.index >= f.train_start) & (df.index < f.cal_start), "demand"])
              for f in config.TEST_FOLDS}
    thresholds = {f.name: df.loc[(df.index >= f.train_start) & (df.index < f.cal_start), "demand"].quantile([1/3, 2/3, .95]).tolist()
                  for f in config.TEST_FOLDS}
    scored = slices(scored, thresholds)
    scored["scaled_ae"] = abs(scored.y - scored.yhat) / scored.fold.map(scales)
    tables = {}
    for label, fields in {"overall": [], "by_fold": ["fold"], "by_season": ["season"], "by_month": ["month"],
                           "by_year_month": ["year_month"], "by_demand": ["demand_regime"],
                           "by_day_night": ["day_night"], "by_week_part": ["week_part"]}.items():
        by = ["model", "regime", "h", *fields]
        table = metrics.score_frame(scored, None, by)
        scaled = scored.groupby(by, observed=True).scaled_ae.mean().rename("mase").reset_index()
        tables[label] = table.merge(scaled, on=by)
        tables[label].to_csv(out / f"{label}.csv", index=False)
    residual_drift(scored, scales).to_csv(out / "residual_drift.csv", index=False)
    if train_weather:
        paired_weather(scored, seed=settings["seed"]).to_csv(out / "paired_weather_vs_ridge.csv", index=False)
    skills = []
    for baseline in ["seasonal_naive_24h", "seasonal_naive_week"]:
        reference = tables["overall"][tables["overall"].model == baseline].set_index("h").mae
        s = tables["overall"][["model", "regime", "h", "mae"]].copy()
        s["baseline"], s["mae_skill"] = baseline, 1 - s.mae / s.h.map(reference)
        skills.append(s)
    pd.concat(skills).to_csv(out / "seasonal_skill.csv", index=False)
    intervals, online_settings = [], {}
    for name in settings["calibration_models"]:
        p = pred[pred.model == name]
        intervals.append(static_intervals(p))
        for mode in ["rolling", "adaptive"]:
            cfg = OnlineConfig(mode=mode, window=settings["calibration_window"], min_scores=settings["calibration_min_scores"],
                               feedback_delay_hours=settings["feedback_delay_hours"], gamma=settings["adaptive_gamma"])
            print(f"Post-hoc intervals: {name}, {mode}", flush=True)
            intervals.append(calibrate_online(p, cfg))
            online_settings[mode] = asdict(cfg)
    all_intervals = pd.concat(intervals, ignore_index=True)
    all_intervals = all_intervals[~all_intervals.target_imputed & all_intervals.y.notna() &
                                   all_intervals.lo.notna() & all_intervals.hi.notna()]
    all_intervals = all_intervals[all_intervals.target_time < all_intervals.fold.map(ends)]
    all_intervals = slices(all_intervals, thresholds)
    all_intervals.to_parquet(out / "intervals.parquet", index=False)
    by = ["model", "regime", "method", "h"]
    interval_summary = interval_table(all_intervals, by)
    interval_summary.to_csv(out / "interval_overall.csv", index=False)
    for label in ["season", "month", "demand_regime", "day_night", "week_part", "fold"]:
        interval_table(all_intervals, [*by, label]).to_csv(out / f"interval_by_{label}.csv", index=False)
    interval_pinball(all_intervals).to_csv(out / "interval_pinball.csv", index=False)
    plots(out, tables["overall"], tables["by_season"], interval_summary)
    for path, expected in original_hashes.items():
        if fingerprint(config.ROOT / path) != expected:
            raise RuntimeError("Historical baseline changed during execution")
    record = {"git": tracking.git_state(), "device": "cpu", "seed": settings["seed"],
              "baseline_seed": old_record["seed"], "bootstrap_seed": settings["seed"],
              "dataset_manifest": tracking.data_manifest(), "processed_sha256": fingerprint(dataset.PROCESSED),
              "baseline_prediction_sha256": original_hashes, "timings": times, "online_config": online_settings,
              "folds": [tracking.fold_dict(f) for f in config.TEST_FOLDS], "mase_scales": scales,
              "demand_thresholds": thresholds, "versions": tracking.versions(), "wall_seconds": time.perf_counter() - start,
              "source_sha256": {str(p.relative_to(config.ROOT)): fingerprint(p) for p in (config.ROOT / "src").rglob("*.py")},
              "model_artifacts": artifacts, "config_sha256": fingerprint(out / "config.json")}
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    print(out, flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=config.ROOT / "configs/weather_uncertainty.json")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), train_weather=not args.analysis_only)

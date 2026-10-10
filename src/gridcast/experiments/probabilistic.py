"""Probabilistic forecast evaluation: raw model quantiles versus conformal calibration.

Reads benchmark predictions; trains nothing. Methods compared per model:

    raw_quantile        quantiles as produced by the model (quantile loss or Gaussian state space)
    cqr_static          conformalised quantile regression, calibrated once per fold
    cqr_rolling         conformalised quantile regression, trailing 60-day online calibration
    conformal_static    point forecast +- calibration-block residual quantile
    conformal_rolling   point forecast +- trailing 60-day residual quantile
    quantile_ensemble   average of the member models' quantiles (then treated like a model)
"""

from __future__ import annotations

import argparse

import pandas as pd

from gridcast import config, tracking
from gridcast.data import dataset
from gridcast.evaluation import backtest, metrics
from gridcast.evaluation.metrics import qcols
from gridcast.probabilistic import conformal

H = config.HEADLINE_HORIZONS
ENSEMBLE_MEMBERS = ["lightgbm_weather", "nbeats_weather", "tcn_weather", "gru_weather"]
POINT_ONLY = ["seasonal_naive_week", "dshw", "ridge_weather"]
OPERATIONAL = ["none", "forecast"]


def quantile_ensemble(pred: pd.DataFrame, members: list[str], name: str = "quantile_ensemble") -> pd.DataFrame:
    sub = pred[pred["model"].isin(members)]
    keys = ["regime", "fold", "origin", "h"]
    cols = ["yhat", *qcols()]
    agg = sub.groupby(keys, observed=True)[cols].mean()
    n = sub.groupby(keys, observed=True)["model"].nunique()
    agg = agg[n == len(set(sub["model"]))].reset_index()
    meta = sub.drop_duplicates(keys)[keys + ["target_time", "y", "target_imputed", "block"]]
    out = agg.merge(meta, on=keys)
    out.insert(0, "model", name)
    return out


def run(experiment: str = "benchmark") -> dict[str, pd.DataFrame]:
    df = dataset.load()
    pred = backtest.load_predictions(experiment)
    pred = pred[pred["h"].isin(H) & pred["regime"].isin(OPERATIONAL) & (pred["model"] != "ercot_day_ahead_reference")]
    has_q = pred.groupby("model")["q0.5"].apply(lambda s: s.notna().all()) if "q0.5" in pred else pd.Series(dtype=bool)
    q_models = [m for m, ok in has_q.items() if ok]
    members = [m for m in ENSEMBLE_MEMBERS if m in q_models]
    frames = []
    if len(members) >= 2:
        ens = quantile_ensemble(pred, members)
        pred = pd.concat([pred, ens], ignore_index=True)
        q_models.append("quantile_ensemble")
    for m in q_models:
        g = pred[pred["model"] == m]
        raw = g[g["block"] == "test"].assign(method="raw_quantile")
        frames += [raw, conformal.calibrate(g, "cqr", "static"), conformal.calibrate(g, "cqr", "rolling")]
    for m in [x for x in POINT_ONLY + ["lightgbm_weather"] if x in set(pred["model"])]:
        g = pred[pred["model"] == m]
        frames += [conformal.calibrate(g, "abs", "static"), conformal.calibrate(g, "abs", "rolling")]
    allp = pd.concat(frames, ignore_index=True)
    allp = allp[~allp["target_imputed"] & allp["y"].notna() & allp[qcols()].notna().all(axis=1)]
    allp = backtest.add_slices(allp)
    scale = backtest.mase_scale(df, allp["origin"].min())
    out = {
        "overall": metrics.score_frame(allp, scale, ["model", "method", "h"]),
        "by_season": metrics.score_frame(allp, scale, ["model", "method", "h", "season"]),
        "by_load_regime": metrics.score_frame(allp, scale, ["model", "method", "h", "load_regime"]),
        "reliability": conformal.reliability(allp, ["model", "method", "h"]),
    }
    for a in config.INTERVAL_ALPHAS:
        c = f"cov{int(round((1 - a) * 100))}"
        out["overall"][f"{c}_error"] = out["overall"][c] - (1 - a)
    with tracking.Run("probabilistic", {"experiment": experiment, "ensemble_members": members}, config.TEST_FOLDS) as r:
        for k, t in out.items():
            t.to_csv(config.TABLES / f"probabilistic_{k}.csv", index=False)
        r.log(n_rows=len(allp))
    # compact interval store for the API / dashboard: the best calibrated variant of the main model
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="benchmark")
    out = run(ap.parse_args(argv).experiment)
    pd.set_option("display.width", 240, "display.max_rows", 300)
    o = out["overall"]
    print(o[["model", "method", "h", "cov50", "cov80", "cov90", "cov95", "width90", "wis", "pinball"]].round(3).to_string())


if __name__ == "__main__":
    main()

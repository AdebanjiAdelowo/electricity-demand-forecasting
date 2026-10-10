"""Turn stored benchmark predictions into result tables (CSV).

Nothing here trains a model. Every number in the documentation is read from
the CSV files this module writes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.data import dataset
from gridcast.evaluation import backtest, metrics

OPERATIONAL = ["none", "forecast"]  # information sets that were really available at the origin


def families(experiment: str) -> dict:
    p = config.TABLES / f"{experiment}_timing.json"
    t = json.loads(p.read_text()) if p.exists() else {}
    return {k: v["family"] for k, v in t.items()}


def load_scored(experiment: str, horizons=None, block: str = "test") -> pd.DataFrame:
    pred = backtest.load_predictions(experiment)
    if horizons is not None:
        pred = pred[pred["h"].isin(horizons)]
    s = backtest.scoreable(pred, block)
    return backtest.add_slices(s)


def _with_family(t: pd.DataFrame, experiment: str) -> pd.DataFrame:
    fam = families(experiment)
    t.insert(1, "family", t["model"].map(fam).fillna("unknown"))
    return t


def common_targets(s: pd.DataFrame) -> pd.DataFrame:
    """Keep (origin, h) pairs that every model/regime with that horizon was able to forecast."""
    key = s["model"] + "|" + s["regime"]
    n_models = s.groupby("h")["model"].transform(lambda x: key.loc[x.index].nunique())
    cnt = s.groupby(["origin", "h"])["yhat"].transform("size")
    return s[cnt == n_models]


def build(experiment: str = "benchmark") -> dict[str, pd.DataFrame]:
    df = dataset.load()
    first_test = pd.read_parquet(next((config.PREDICTIONS / experiment).glob("*.parquet")), columns=["origin", "block"])
    scale = backtest.mase_scale(df, first_test.loc[first_test.block == "test", "origin"].min())
    dense = load_scored(experiment)
    ref = dense[dense.model == "ercot_day_ahead_reference"]
    dense = dense[dense.model != "ercot_day_ahead_reference"]
    s = common_targets(dense[dense["h"].isin(config.HEADLINE_HORIZONS)])
    out = {}

    overall = _with_family(metrics.score_frame(s, scale, ["model", "regime", "h"]), experiment)
    out["overall"] = overall.sort_values(["h", "mae"])
    out["by_season"] = _with_family(metrics.score_frame(s, scale, ["model", "regime", "h", "season"]), experiment)
    out["by_load_regime"] = _with_family(metrics.score_frame(s, scale, ["model", "regime", "h", "load_regime"]), experiment)
    out["by_fold"] = _with_family(metrics.score_frame(s, scale, ["model", "regime", "h", "fold"]), experiment)
    out["peak_hour"] = _with_family(backtest.daily_peak_errors(s), experiment)
    out["horizon_profile"] = _with_family(metrics.score_frame(dense, scale, ["model", "regime", "h"]), experiment)

    # significance: block-bootstrap CI of the MAE difference to the best non-deep operational model per horizon
    op = s[s["regime"].isin(OPERATIONAL)]
    fam = families(experiment)
    boots = []
    for _h, g in op.groupby("h"):
        cand = g[g["model"].map(fam).isin(["statistical", "ml"])]
        best = cand.assign(ae=(cand.y - cand.yhat).abs()).groupby("model")["ae"].mean().idxmin()
        b = backtest.paired_bootstrap(g.assign(regime="operational"), best)
        b["baseline"] = best
        boots.append(b)
    out["bootstrap_vs_best_classical"] = _with_family(pd.concat(boots, ignore_index=True), experiment)

    if len(ref):
        # ERCOT's published day-ahead forecast, scored on the same target hours as our 24 h forecasts
        r = ref.dropna(subset=["yhat"])
        ours = s[(s.h == 24) & s["regime"].isin(OPERATIONAL)]
        keep = set(r["target_time"]) & set(ours["target_time"])
        both = pd.concat([r[r["target_time"].isin(keep)], ours[ours["target_time"].isin(keep)]])
        out["vs_ercot_day_ahead"] = metrics.score_frame(both, scale, ["model", "regime"]).sort_values("mae")

    edges = backtest.load_regime_edges(s)
    out["meta"] = pd.DataFrame([{"experiment": experiment, "mase_scale_mw": scale, "n_common_targets_per_h": int(s.groupby(["model", "regime", "h"]).size().min()),
                                 "test_start": str(s["origin"].min()), "test_end": str(s["origin"].max()),
                                 "mean_load_mw": float(s.drop_duplicates("target_time")["y"].mean()), **{f"load_{k}": v for k, v in edges.items()}}])
    for name, t in out.items():
        t.to_csv(config.TABLES / f"{experiment}_{name}.csv", index=False)
    return out


def compute_table(experiment: str = "benchmark") -> pd.DataFrame:
    t = json.loads((config.TABLES / f"{experiment}_timing.json").read_text())
    overall = pd.read_csv(config.TABLES / f"{experiment}_overall.csv")
    op = overall[overall["regime"].isin(OPERATIONAL)]
    mae = op.pivot_table(index="model", columns="h", values="mae")
    rows = []
    for name, v in t.items():
        if name not in mae.index:
            continue
        rows.append({"model": name, "family": v["family"], "strategy": v["strategy"], "n_params": v["n_params"],
                     "fit_seconds_per_fold": round(v["fit_seconds_per_fold"], 2),
                     "inference_ms_per_1000_forecasts": round(v["inference_ms_per_1000_forecasts"], 3),
                     "device": (v.get("extra", {}).get("test1", {}) or {}).get("device", "cpu"),
                     **{f"mae_h{h}": float(mae.loc[name, h]) for h in mae.columns}})
    out = pd.DataFrame(rows).sort_values("fit_seconds_per_fold")
    out["mean_mae_headline"] = out[[c for c in out.columns if c.startswith("mae_h")]].mean(axis=1)
    cost = out["fit_seconds_per_fold"].to_numpy()
    err = out["mean_mae_headline"].to_numpy()
    out["pareto_optimal"] = [not np.any((cost <= cost[i]) & (err < err[i]) & (np.arange(len(out)) != i)) for i in range(len(out))]
    out.to_csv(config.TABLES / f"{experiment}_compute.csv", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="benchmark")
    a = ap.parse_args(argv)
    out = build(a.experiment)
    pd.set_option("display.width", 220)
    o = out["overall"]
    print(o[o["regime"].isin(OPERATIONAL)].pivot_table(index=["family", "model"], columns="h", values="mae").round(0))
    print(o[o["regime"] == "oracle"].pivot_table(index=["family", "model"], columns="h", values="mae").round(0))
    if "vs_ercot_day_ahead" in out:
        print(out["vs_ercot_day_ahead"][["model", "regime", "n", "mae", "mape", "bias"]].round(2))
    print(compute_table(a.experiment).round(1).to_string())


if __name__ == "__main__":
    main()

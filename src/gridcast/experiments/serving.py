"""Build the compact artefacts that the API, dashboard and copilot read.

* results/serving/forecasts.parquet   test-block forecasts at the headline horizons with
                                      rolling-conformal 90 percent intervals, for a few models
* results/tables/feature_importance.csv  LightGBM gain shares per horizon (last test fold)
"""

from __future__ import annotations

import pandas as pd

from gridcast import config
from gridcast.data import dataset
from gridcast.evaluation import backtest
from gridcast.experiments import zoo
from gridcast.models.tabular import LightGBMForecaster
from gridcast.probabilistic import conformal

H = config.HEADLINE_HORIZONS
SERVED = {  # model -> (regime, conformal kind)
    "lightgbm_weather": ("forecast", "cqr"),
    "nbeats_weather": ("forecast", "cqr"),
    "lightgbm": ("none", "cqr"),
    "ridge_weather": ("forecast", "abs"),
    "dshw": ("none", "abs"),
    "seasonal_naive_week": ("none", "abs"),
}
GROUPS = {"l_": "lagged demand", "c_": "calendar", "w_": "weather"}


def forecasts(experiment: str = "benchmark") -> pd.DataFrame:
    out = []
    have = {p.stem for p in (config.PREDICTIONS / experiment).glob("*.parquet")}
    for model, (regime, kind) in SERVED.items():
        if model not in have:
            continue
        p = backtest.load_predictions(experiment, [model])
        p = p[p["h"].isin(H) & (p["regime"] == regime)]
        c = conformal.calibrate(p, kind, "rolling")
        out.append(c[["model", "regime", "method", "fold", "origin", "h", "target_time", "y", "target_imputed", "yhat", "q0.05", "q0.5", "q0.95"]])
    f = pd.concat(out, ignore_index=True)
    f.loc[f["target_imputed"], "y"] = float("nan")
    f = f.drop(columns="target_imputed")
    for c in ("y", "yhat", "q0.05", "q0.5", "q0.95"):
        f[c] = f[c].round(1).astype("float32")
    d = config.RESULTS / "serving"
    d.mkdir(parents=True, exist_ok=True)
    f.to_parquet(d / "forecasts.parquet", index=False)
    return f


def feature_importance() -> pd.DataFrame:
    df = dataset.load()
    m = LightGBMForecaster(zoo.W, "lightgbm_weather", quantiles=False)
    m.fit(df, config.TEST_FOLDS[-1], H)
    rows = []
    for h in H:
        s = m.importance(h)
        for feat, v in s.items():
            rows.append({"h": h, "feature": feat, "group": GROUPS.get(feat[:2], "other"), "gain_share": float(v)})
    t = pd.DataFrame(rows)
    t.to_csv(config.TABLES / "feature_importance.csv", index=False)
    return t


def main(argv=None):
    f = forecasts()
    print(f.groupby(["model", "regime", "method"]).size())
    t = feature_importance()
    print(t.groupby(["h", "group"])["gain_share"].sum().unstack().round(3))


if __name__ == "__main__":
    main()

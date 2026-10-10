"""Python tools the copilot is allowed to call.

Every number the copilot reports comes from one of these functions. They read
the stored result tables, the stored backtest forecasts and the processed
demand series. None of them trains a model, changes a file or talks to any
grid equipment. Each result names the file it was read from.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from gridcast import config
from gridcast.data import dataset

SERVING = config.RESULTS / "serving"
MAX_ROWS = 48


def _rel(path) -> str:
    return str(path.relative_to(config.ROOT))


def _r(x, nd: int = 1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


@lru_cache(maxsize=1)
def _history() -> pd.DataFrame:
    return dataset.load()


@lru_cache(maxsize=8)
def _table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.TABLES / f"{name}.csv")


@lru_cache(maxsize=1)
def _forecasts() -> pd.DataFrame:
    return pd.read_parquet(SERVING / "forecasts.parquet")


def _ts(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def list_models() -> dict:
    """Models with stored results and the information each one used."""
    t = _table("benchmark_overall")
    rows = t.groupby(["model", "family", "regime"], as_index=False).size().drop(columns="size")
    served = sorted(_forecasts()["model"].unique()) if (SERVING / "forecasts.parquet").exists() else []
    return {"models": rows.to_dict("records"), "models_with_stored_forecasts": served,
            "regimes": {"none": "no weather inputs", "forecast": "archived, bias-corrected NWP forecasts (operational)",
                        "forecast_raw": "archived NWP forecasts without bias correction (ablation)",
                        "oracle": "observed weather at the target hour (NOT available in practice; upper bound)"},
            "source": _rel(config.TABLES / "benchmark_overall.csv")}


def query_history(start: str, end: str, aggregate: str = "summary") -> dict:
    """Observed ERCOT demand between two UTC timestamps (hour ending). aggregate: summary | daily | hourly."""
    df = _history()
    a, b = _ts(start), _ts(end)
    s = df.loc[(df.index >= a) & (df.index <= b)]
    if s.empty:
        return {"error": f"no data between {a} and {b}; data covers {df.index.min()} to {df.index.max()}"}
    y = s["demand_raw"]
    out = {"start": str(s.index.min()), "end": str(s.index.max()), "unit": "MW (hourly average demand)", "hours": int(len(s)),
           "missing_hours": int(y.isna().sum()), "mean_mw": _r(y.mean()), "min_mw": _r(y.min()), "max_mw": _r(y.max()),
           "peak_time_utc": str(y.idxmax()), "peak_time_local": str(y.idxmax().tz_convert(config.LOCAL_TZ)),
           "mean_temperature_c": _r(s["temp"].mean(), 2), "source": _rel(dataset.PROCESSED)}
    if aggregate == "daily":
        local = s.set_index(s.index.tz_convert(config.LOCAL_TZ) - pd.Timedelta(hours=1))
        d = local["demand_raw"].resample("D").agg(["mean", "max", "min"]).round(1).tail(MAX_ROWS)
        out["daily_local"] = [{"date": str(i.date()), "mean_mw": r["mean"], "max_mw": r["max"], "min_mw": r["min"]} for i, r in d.iterrows()]
    elif aggregate == "hourly":
        out["hourly"] = [{"utc_end": str(i), "demand_mw": _r(v)} for i, v in y.tail(MAX_ROWS).items()]
    return out


def get_forecast(target_time: str, horizon: int = 24, model: str | None = None) -> dict:
    """Stored backtest forecast for the hour ending at target_time (UTC), issued `horizon` hours earlier."""
    f = _forecasts()
    t = _ts(target_time).floor("h")
    model = model or f["model"].iloc[0]
    g = f[(f["target_time"] == t) & (f["h"] == horizon) & (f["model"] == model)]
    if g.empty:
        return {"error": "no stored forecast for that target, horizon and model",
                "available_models": sorted(f["model"].unique()), "available_horizons": sorted(int(h) for h in f["h"].unique()),
                "available_targets": [str(f["target_time"].min()), str(f["target_time"].max())]}
    r = g.iloc[0]
    out = {"model": model, "weather_information": r["regime"], "horizon_hours": int(horizon),
           "issued_at_utc": str(r["origin"]), "target_hour_ending_utc": str(t),
           "target_hour_ending_local": str(t.tz_convert(config.LOCAL_TZ)), "forecast_mw": _r(r["yhat"]),
           "interval_90_low_mw": _r(r["q0.05"]), "interval_90_high_mw": _r(r["q0.95"]),
           "interval_method": r["method"], "actual_mw": _r(r["y"]),
           "note": "This is a stored out-of-sample backtest forecast, not a live forecast.",
           "source": _rel(SERVING / "forecasts.parquet")}
    if out["actual_mw"] is not None:
        out["error_mw"] = _r(r["yhat"] - r["y"])
        out["absolute_percentage_error"] = _r(100 * abs(r["yhat"] - r["y"]) / r["y"], 2)
        out["actual_inside_90_interval"] = bool(r["q0.05"] <= r["y"] <= r["q0.95"])
    return out


def compare_models(horizon: int = 24, metric: str = "mae", by: str = "overall", include_oracle: bool = False) -> dict:
    """Rank models at one horizon on the test period. by: overall | season | load_regime."""
    name = {"overall": "benchmark_overall", "season": "benchmark_by_season", "load_regime": "benchmark_by_load_regime"}.get(by)
    if name is None:
        return {"error": "by must be one of overall, season, load_regime"}
    t = _table(name)
    if metric not in t.columns:
        return {"error": f"unknown metric {metric}", "available": [c for c in ("mae", "rmse", "mape", "smape", "mase", "bias") if c in t.columns]}
    t = t[t["h"] == horizon]
    if t.empty:
        return {"error": f"no results for horizon {horizon}", "available_horizons": sorted(int(h) for h in _table(name)["h"].unique())}
    regimes = ["none", "forecast"] + (["oracle"] if include_oracle else [])
    t = t[t["regime"].isin(regimes)]
    cols = ["model", "family", "regime"] + ([by] if by != "overall" else []) + [metric, "n"]
    key = [by, metric] if by != "overall" else [metric]
    rows = t.sort_values(key, key=(lambda s: s.abs()) if metric == "bias" else None)[cols]
    rows[metric] = rows[metric].round(3 if metric in ("mase", "mape", "smape") else 1)
    meta = _table("benchmark_meta").iloc[0]
    return {"horizon_hours": horizon, "metric": metric, "unit": "MW" if metric in ("mae", "rmse", "bias") else ("percent" if "ape" in metric else "ratio"),
            "test_period_utc": [meta["test_start"], meta["test_end"]], "ranking": rows.to_dict("records"),
            "note": "Regime 'oracle' uses observed future weather and is not an operational result." if include_oracle else
                    "Only operational information sets (no weather, or archived weather forecasts) are ranked.",
            "source": _rel(config.TABLES / f"{name}.csv")}


def calibration(model: str = "lightgbm_weather", horizon: int = 24) -> dict:
    """Prediction-interval coverage and width for each uncertainty method of one model."""
    t = _table("probabilistic_overall")
    g = t[(t["model"] == model) & (t["h"] == horizon)]
    if g.empty:
        return {"error": "no probabilistic results for that model and horizon", "available_models": sorted(t["model"].unique())}
    rows = [{"method": r["method"], "coverage_of_90_interval": _r(r["cov90"], 3), "coverage_of_50_interval": _r(r["cov50"], 3),
             "mean_width_90_mw": _r(r["width90"]), "wis_mw": _r(r["wis"]), "pinball_mw": _r(r["pinball"])} for _, r in g.iterrows()]
    return {"model": model, "horizon_hours": horizon, "nominal": {"coverage_of_90_interval": 0.9, "coverage_of_50_interval": 0.5},
            "methods": rows, "source": _rel(config.TABLES / "probabilistic_overall.csv")}


def feature_importance(horizon: int = 24, top: int = 10) -> dict:
    """Gain-based importance of the LightGBM weather model's inputs, grouped and per feature."""
    t = _table("feature_importance")
    g = t[t["h"] == horizon].sort_values("gain_share", ascending=False)
    if g.empty:
        return {"error": f"no importance stored for horizon {horizon}", "available_horizons": sorted(int(h) for h in t["h"].unique())}
    groups = g.groupby("group")["gain_share"].sum().sort_values(ascending=False)
    return {"model": "lightgbm_weather", "horizon_hours": horizon,
            "group_gain_share": {k: _r(v, 3) for k, v in groups.items()},
            "top_features": [{"feature": r["feature"], "group": r["group"], "gain_share": _r(r["gain_share"], 3)} for _, r in g.head(top).iterrows()],
            "caveat": "Split-gain importance describes what the fitted trees used. It is not a causal attribution, "
                      "and correlated inputs share credit arbitrarily. See the ablation study for measured effects.",
            "source": _rel(config.TABLES / "feature_importance.csv")}


def unusual_observations(start: str | None = None, end: str | None = None, top: int = 10) -> dict:
    """Hours flagged as unusual by the seasonal-residual detector. Flags are unverified."""
    from gridcast.anomaly import detectors as det

    df = _history()
    score = det.score_robust_seasonal(df)
    resid = det.seasonal_residual(df["demand"])
    thr = float(score.quantile(0.995))
    m = pd.Series(True, index=df.index)
    if start:
        m &= df.index >= _ts(start)
    if end:
        m &= df.index <= _ts(end)
    s = score[m & (score > thr)].sort_values(ascending=False).head(top)
    rows = [{"utc_end": str(i), "local": str(i.tz_convert(config.LOCAL_TZ)), "demand_mw": _r(df.at[i, "demand"]),
             "deviation_from_recent_same_hour_pct": _r(100 * resid[i], 1), "score": _r(v, 1)} for i, v in s.items()]
    return {"detector": "robust seasonal residual (median of the same hour in the previous 4 weeks, trailing MAD scale)",
            "threshold_score": _r(thr, 2), "flagged_hours_in_window": int((score[m] > thr).sum()), "hours_in_window": int(m.sum()),
            "top": rows, "status": "Unverified statistical flags. A flag is not evidence of a grid event or a data error.",
            "source": _rel(dataset.PROCESSED)}


TOOLS = {
    "list_models": list_models,
    "query_history": query_history,
    "get_forecast": get_forecast,
    "compare_models": compare_models,
    "calibration": calibration,
    "feature_importance": feature_importance,
    "unusual_observations": unusual_observations,
}


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


_STR, _INT, _BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}

TOOL_SCHEMAS = [
    {"name": "list_models", "description": "List the forecasting models that have stored results, and what each weather-information regime means. Call this when unsure which model names exist.",
     "input_schema": _schema({}, [])},
    {"name": "query_history", "description": "Observed ERCOT hourly demand statistics between two UTC timestamps (ISO 8601). Use for any question about what demand actually was.",
     "input_schema": _schema({"start": _STR, "end": _STR, "aggregate": {"type": "string", "enum": ["summary", "daily", "hourly"]}}, ["start", "end", "aggregate"])},
    {"name": "get_forecast", "description": "Retrieve one stored out-of-sample backtest forecast with its 90 percent interval and the realised value. target_time is the UTC hour-ending timestamp being forecast; horizon is the lead time in hours (1, 24, 48 or 168).",
     "input_schema": _schema({"target_time": _STR, "horizon": _INT, "model": _STR}, ["target_time", "horizon", "model"])},
    {"name": "compare_models", "description": "Rank models at one horizon on the test period by an error metric (mae, rmse, mape, smape, mase, bias), overall or split by season or load regime.",
     "input_schema": _schema({"horizon": _INT, "metric": _STR, "by": {"type": "string", "enum": ["overall", "season", "load_regime"]}, "include_oracle": _BOOL}, ["horizon", "metric", "by", "include_oracle"])},
    {"name": "calibration", "description": "Empirical coverage and width of prediction intervals for each uncertainty method of one model at one horizon.",
     "input_schema": _schema({"model": _STR, "horizon": _INT}, ["model", "horizon"])},
    {"name": "feature_importance", "description": "Which inputs the LightGBM weather model relied on at one horizon (gain share by group and by feature). Descriptive, not causal.",
     "input_schema": _schema({"horizon": _INT, "top": _INT}, ["horizon", "top"])},
    {"name": "unusual_observations", "description": "Hours of observed demand flagged as statistically unusual in a UTC window. Flags are unverified and must be reported as such.",
     "input_schema": _schema({"start": _STR, "end": _STR, "top": _INT}, ["start", "end", "top"])},
]


def call(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}", "available": list(TOOLS)}
    try:
        return TOOLS[name](**args)
    except FileNotFoundError as e:
        return {"error": f"results are not built yet: {e}"}
    except (TypeError, ValueError, KeyError) as e:
        return {"error": f"{type(e).__name__}: {e}"}

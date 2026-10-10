"""Read-only HTTP API over the study's stored results.

    uvicorn gridcast.api.main:app --port 8000

Every endpoint delegates to the same Python tools the copilot uses, so the
API, the dashboard and the copilot can never disagree about a number. The
forecasts served are stored out-of-sample backtest forecasts, not live ones.
"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from gridcast import config
from gridcast.copilot import agent, tools

app = FastAPI(
    title="ERCOT demand forecasting study API",
    version="0.1.0",
    description="Read-only access to observed demand, stored backtest forecasts, model comparisons, "
                "interval calibration, anomaly flags and the Grid Copilot. Not an operational system.",
)


def _ok(result: dict) -> dict:
    if "error" in result:
        raise HTTPException(status_code=404, detail=result)
    return result


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "processed_data": (config.DATA_PROCESSED / "ercot_hourly.parquet").exists(),
            "forecasts": (config.RESULTS / "serving" / "forecasts.parquet").exists(),
            "copilot_backend": "llm" if agent.credentials_available() else "offline"}


@app.get("/meta")
def meta() -> dict:
    p = config.DATA_PROCESSED / "manifest.json"
    man = json.loads(p.read_text()) if p.exists() else {}
    return {"region": config.REGION_NAME, "time_index": "UTC, hour ending", "local_time_zone": config.LOCAL_TZ,
            "horizons_hours": config.HEADLINE_HORIZONS, "test_period_utc": [str(config.TEST_START), str(config.TEST_END)],
            "dataset": {k: man.get(k) for k in ("dataset_version", "rows", "start", "end", "licenses")}}


@app.get("/models")
def models() -> dict:
    return _ok(tools.call("list_models", {}))


@app.get("/history")
def history(start: str, end: str, aggregate: Literal["summary", "daily", "hourly"] = "summary") -> dict:
    return _ok(tools.call("query_history", {"start": start, "end": end, "aggregate": aggregate}))


@app.get("/forecast")
def forecast(target_time: str, horizon: int = Query(24, description="lead time in hours: 1, 24, 48 or 168"),
             model: str | None = None) -> dict:
    return _ok(tools.call("get_forecast", {"target_time": target_time, "horizon": horizon, "model": model}))


@app.get("/compare")
def compare(horizon: int = 24, metric: str = "mae", by: Literal["overall", "season", "load_regime"] = "overall",
            include_oracle: bool = False) -> dict:
    return _ok(tools.call("compare_models", {"horizon": horizon, "metric": metric, "by": by, "include_oracle": include_oracle}))


@app.get("/calibration")
def calibration(model: str = "lightgbm_weather", horizon: int = 24) -> dict:
    return _ok(tools.call("calibration", {"model": model, "horizon": horizon}))


@app.get("/feature-importance")
def feature_importance(horizon: int = 24, top: int = 10) -> dict:
    return _ok(tools.call("feature_importance", {"horizon": horizon, "top": top}))


@app.get("/anomalies")
def anomalies(start: str | None = None, end: str | None = None, top: int = 10) -> dict:
    return _ok(tools.call("unusual_observations", {"start": start, "end": end, "top": top}))


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    backend: Literal["auto", "llm", "offline"] = "auto"


@app.post("/copilot/ask")
def copilot_ask(q: Question) -> dict:
    try:
        return agent.ask(q.question, q.backend).to_dict()
    except Exception as e:  # missing credentials or an upstream API failure
        raise HTTPException(status_code=503, detail=f"copilot backend unavailable: {type(e).__name__}: {e}") from e

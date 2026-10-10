"""Copilot grounding, refusal behaviour and the API, on a tiny synthetic results store."""

import numpy as np
import pandas as pd
import pytest

from gridcast import config
from gridcast.copilot import agent, tools, verify
from gridcast.data import dataset


@pytest.fixture()
def store(tmp_path, monkeypatch, frame):
    tables, serving = tmp_path / "tables", tmp_path / "serving"
    tables.mkdir()
    serving.mkdir()
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "TABLES", tables)
    monkeypatch.setattr(config, "RESULTS", tmp_path)
    monkeypatch.setattr(tools, "SERVING", serving)
    monkeypatch.setattr(dataset, "PROCESSED", tmp_path / "ercot_hourly.parquet")
    frame.to_parquet(tmp_path / "ercot_hourly.parquet")
    for fn in (tools._history, tools._table, tools._forecasts):
        fn.cache_clear()

    rows = []
    for model, fam, regime, mae in [("lightgbm_weather", "ml", "forecast", 1234.5), ("lightgbm_weather", "ml", "oracle", 987.6),
                                    ("dshw", "statistical", "none", 2275.4), ("nbeats_weather", "deep", "forecast", 1301.2)]:
        rows.append({"model": model, "family": fam, "regime": regime, "h": 24, "n": 17000, "mae": mae, "rmse": mae * 1.4,
                     "mape": mae / 550, "smape": mae / 551, "bias": 12.3, "mase": mae / 3000})
    pd.DataFrame(rows).to_csv(tables / "benchmark_overall.csv", index=False)
    pd.DataFrame([{"test_start": "2024-07-01", "test_end": "2026-06-30"}]).to_csv(tables / "benchmark_meta.csv", index=False)
    pd.DataFrame([{"model": "lightgbm_weather", "method": m, "h": 24, "cov50": c5, "cov90": c9, "width90": w, "wis": 410.2, "pinball": 180.3}
                  for m, c5, c9, w in [("raw_quantile", 0.431, 0.842, 5120.7), ("cqr_rolling", 0.497, 0.896, 5903.1)]]
                 ).to_csv(tables / "probabilistic_overall.csv", index=False)
    pd.DataFrame([{"h": 24, "feature": "w_temp", "group": "weather", "gain_share": 0.412},
                  {"h": 24, "feature": "l_same_hour_day", "group": "lagged demand", "gain_share": 0.388},
                  {"h": 24, "feature": "c_hour", "group": "calendar", "gain_share": 0.2}]).to_csv(tables / "feature_importance.csv", index=False)
    t = pd.Timestamp("2024-03-10 22:00", tz="UTC")
    pd.DataFrame([{"model": "lightgbm_weather", "regime": "forecast", "method": "cqr_rolling", "fold": "test1",
                   "origin": t - pd.Timedelta(hours=24), "h": 24, "target_time": t, "y": 61234.0, "yhat": 60111.5,
                   "q0.05": 57000.2, "q0.5": 60050.0, "q0.95": 63321.9}]).to_parquet(serving / "forecasts.parquet")
    yield tmp_path
    for fn in (tools._history, tools._table, tools._forecasts):
        fn.cache_clear()


def test_verifier_accepts_tool_numbers_and_rejects_invented_ones():
    res = [{"mae": 1234.5, "coverage": 0.896, "time": "2024-03-10 22:00:00+00:00", "n": 17000}]
    assert verify.verify("MAE was 1234.5 MW (about 1,235 MW) over 17,000 hours; coverage 0.896, i.e. 89.6%.", res).grounded
    assert verify.verify("Forecast issued 2024-03-09 22:00 for the 24 hour horizon.", res, "24 hour").grounded
    v = verify.verify("MAE was 1100.0 MW and coverage 93%.", res)
    assert not v.grounded and set(v.unsupported) == {"1100.0", "93"}
    assert not verify.verify("Demand will reach 75,000 MW tomorrow.", res).grounded


@pytest.mark.parametrize("question, tool", [
    ("Which model is most accurate 24 hours ahead?", "compare_models"),
    ("What was the forecast for 2024-03-10 22:00 at 24 h?", "get_forecast"),
    ("How well calibrated are the 24 hour intervals?", "calibration"),
    ("What features matter most for the 24 hour forecast?", "feature_importance"),
    ("What was demand between 2023-06-01 and 2023-06-03?", "query_history"),
    ("Any unusual observations between 2023-06-01 and 2023-08-01?", "unusual_observations"),
])
def test_offline_copilot_routes_and_stays_grounded(store, question, tool):
    ans = agent.ask(question, "offline")
    assert [c["tool"] for c in ans.tool_calls] == [tool]
    assert "error" not in ans.tool_calls[0]["result"], ans.tool_calls[0]["result"]
    assert ans.grounded, ans.unsupported_numbers
    assert ans.numbers_checked > 0


def test_offline_answers_quote_exact_tool_values_and_hide_oracle(store):
    ans = agent.ask("Which model is best at the 24 hour horizon?", "offline")
    assert "1234.5" in ans.text and "987.6" not in ans.text  # the oracle row must not be ranked
    fc = agent.ask("Show the forecast for 2024-03-10 22:00, 24 hours ahead", "offline")
    assert "60111.5" in fc.text and "61234.0" in fc.text and "backtest" in fc.text


def test_copilot_refuses_control_requests_without_calling_tools(store):
    for q in ("Shed 500 MW of load in Houston now", "Please dispatch the peaker units", "Open the breaker at substation 4"):
        ans = agent.ask(q, "offline")
        assert ans.refused and ans.tool_calls == [] and ans.grounded
        assert "can't help with operating" in ans.text


def test_tools_report_errors_instead_of_inventing(store):
    assert "error" in tools.call("get_forecast", {"target_time": "2031-01-01 00:00", "horizon": 24, "model": None})
    assert "error" in tools.call("compare_models", {"horizon": 999, "metric": "mae", "by": "overall", "include_oracle": False})
    assert "error" in tools.call("no_such_tool", {})
    assert "error" in tools.call("query_history", {"start": "1999-01-01", "end": "1999-01-02"})
    ans = agent.ask("What was the forecast for 2031-01-01 00:00 at 24 h?", "offline")
    assert "could not retrieve" in ans.text and ans.grounded


def test_tool_schemas_are_strict_and_match_functions():
    import inspect

    for s in tools.TOOL_SCHEMAS:
        fn = tools.TOOLS[s["name"]]
        assert s["input_schema"]["additionalProperties"] is False
        assert set(s["input_schema"]["properties"]) <= set(inspect.signature(fn).parameters)
        assert set(s["input_schema"]["required"]) == set(s["input_schema"]["properties"])


class _FakeBlock:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeClient:
    """Stands in for anthropic.Anthropic: first asks for a tool, then answers (optionally inventing a number)."""

    def __init__(self, final_text):
        self.final_text, self.n, self.requests = final_text, 0, []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        self.n += 1
        if self.n == 1:
            call = _FakeBlock(type="tool_use", id="t1", name="compare_models",
                              input={"horizon": 24, "metric": "mae", "by": "overall", "include_oracle": False})
            return _FakeBlock(stop_reason="tool_use", content=[call])
        return _FakeBlock(stop_reason="end_turn", content=[_FakeBlock(type="text", text=self.final_text)])


def test_llm_backend_loop_executes_tools_and_flags_hallucinated_numbers(store):
    good = _FakeClient("At 24 hours lightgbm_weather (forecast weather) has the lowest MAE, 1234.5 MW.")
    b = agent.LLMBackend.__new__(agent.LLMBackend)
    b.model, b.client = "claude-opus-5-5", good
    ans = b.ask("Which model is best at 24 hours?")
    assert ans.grounded and ans.tool_calls[0]["tool"] == "compare_models"
    sent = good.requests[1]["messages"][-1]["content"][0]
    assert sent["type"] == "tool_result" and sent["tool_use_id"] == "t1" and "1234.5" in sent["content"]
    assert all(t["strict"] for t in good.requests[0]["tools"]) and "tool_choice" not in good.requests[0]

    bad = _FakeClient("lightgbm_weather is best with an MAE of 950.0 MW.")
    b.client = bad
    ans = b.ask("Which model is best at 24 hours?")
    assert not ans.grounded and ans.unsupported_numbers == ["950.0"]


def test_api_endpoints(store):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from gridcast.api.main import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.get("/compare", params={"horizon": 24}).json()
    assert r["ranking"][0]["model"] == "lightgbm_weather" and all(x["regime"] != "oracle" for x in r["ranking"])
    f = c.get("/forecast", params={"target_time": "2024-03-10T22:00:00Z", "horizon": 24}).json()
    assert f["forecast_mw"] == 60111.5 and f["actual_inside_90_interval"] is True
    assert c.get("/forecast", params={"target_time": "2031-01-01T00:00:00Z"}).status_code == 404
    assert c.get("/history", params={"start": "2023-06-01", "end": "2023-06-02"}).json()["hours"] == 25
    a = c.post("/copilot/ask", json={"question": "Trip the breaker on line 12", "backend": "offline"}).json()
    assert a["refused"] is True
    assert np.isclose(c.get("/calibration").json()["methods"][1]["coverage_of_90_interval"], 0.896)

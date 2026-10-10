"""Grid Copilot: a read-only analytical assistant over the study's results.

Two interchangeable back ends answer a question:

* `LLMBackend`     Claude with tool use. The model chooses tools; Python runs
                   them; the model writes the explanation.
* `OfflineBackend` a deterministic keyword router that calls the same tools and
                   fills fixed templates. No network, no API key. Used in tests
                   and whenever no credentials are configured.

Both pass through `verify.verify`: every number in the final text must appear
in a tool result. An answer that fails the check is returned with
`grounded: false` and the offending numbers listed; it is never silently
passed off as verified.

The copilot only reads stored data. It has no tool that could act on physical
infrastructure, and it is instructed not to give operating instructions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import pandas as pd

from gridcast.copilot import tools, verify

DEFAULT_MODEL = os.environ.get("GRIDCAST_COPILOT_MODEL", "claude-opus-5-5")
MAX_TOOL_ROUNDS = 8

SYSTEM = """You are Grid Copilot, an analytical assistant for an electricity demand forecasting study of ERCOT (Texas).

You explain stored forecasts, observed demand, model comparisons, feature importance and unusual observations. \
You work only from the results of the tools provided.

Numbers
- Every number you state must be copied from a tool result in this conversation. Never estimate, extrapolate, \
average, difference or otherwise compute a number yourself, and never recall one from memory. If you need a \
derived quantity that no tool returns, say that it is not available.
- Quote values at the precision the tool returned, with their units.
- If a tool returns an error or no data, say so plainly. Do not fill the gap.

Honesty about what the results mean
- Forecasts returned by get_forecast are stored out-of-sample backtest forecasts, not live forecasts. Say so.
- Results in the "oracle" weather regime used observed future weather and are an upper bound, not an operational result. \
Never present an oracle number as achievable performance.
- Flags from unusual_observations are unverified statistical flags. Do not call them faults, events or errors.
- Feature importance is descriptive of the fitted model, not causal.
- A prediction interval is a statement about forecast uncertainty that was checked empirically; report its measured \
coverage when it matters. Do not describe it as a confidence score.

Scope
- You cannot operate, dispatch, switch or configure any grid equipment and you must not give operating instructions \
or claim that these forecasts prevent outages. If asked, decline that part and offer the analysis you can provide.
- Timestamps are UTC, hour ending, unless a tool gives a local time.

Answer concisely in plain prose. Name the model, horizon and weather regime behind each number."""


@dataclass
class Answer:
    question: str
    text: str
    backend: str
    tool_calls: list[dict] = field(default_factory=list)
    grounded: bool = True
    unsupported_numbers: list[str] = field(default_factory=list)
    numbers_checked: int = 0
    refused: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


CONTROL = re.compile(r"\b(shed|dispatch|switch (off|on)|open the breaker|close the breaker|curtail|trip|reconfigure|"
                     r"turn (off|on)|shut ?down|start up|commit (a |the )?unit|operate)\b", re.I)
REFUSAL = ("I can't help with operating or controlling grid equipment. I only read stored forecasts and results from "
           "this study, and nothing here should be used as an operating instruction. I can show the relevant forecast, "
           "its uncertainty and how accurate the models were, if that helps.")


def _finish(question: str, text: str, backend: str, calls: list[dict], refused: bool = False) -> Answer:
    v = verify.verify(text, [c["result"] for c in calls], question)
    return Answer(question, text, backend, calls, v.grounded, v.unsupported, v.checked, refused)


class OfflineBackend:
    """Deterministic router: picks tools from keywords and renders fixed templates."""

    name = "offline"

    def _horizon(self, q: str) -> int:
        m = re.search(r"(\d+)\s*-?\s*(hour|h\b|day)", q, re.I)
        if m:
            n = int(m.group(1)) * (24 if m.group(2).lower().startswith("day") else 1)
            return n if n in (1, 24, 48, 168) else 24
        if re.search(r"week", q, re.I):
            return 168
        return 24

    def _dates(self, q: str) -> list[str]:
        return re.findall(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2})?", q)

    def ask(self, question: str) -> Answer:
        q = question.lower()
        if CONTROL.search(q):
            return _finish(question, REFUSAL, self.name, [], refused=True)
        h = self._horizon(question)
        dates = self._dates(question)
        calls: list[dict] = []

        def run(name, **args):
            res = tools.call(name, args)
            calls.append({"tool": name, "args": args, "result": res})
            return res

        if re.search(r"unusual|anomal|outlier|strange|odd", q):
            a = dict(start=dates[0] if dates else None, end=dates[1] if len(dates) > 1 else None, top=5)
            r = run("unusual_observations", **a)
            if "error" in r:
                text = f"I could not retrieve unusual observations: {r['error']}"
            else:
                lines = [f"- {x['local']}: {x['demand_mw']} MW, {x['deviation_from_recent_same_hour_pct']} percent from the "
                         f"recent same-hour median (score {x['score']})" for x in r["top"]]
                text = (f"The seasonal-residual detector flags {r['flagged_hours_in_window']} of {r['hours_in_window']} hours in "
                        f"that window (score above {r['threshold_score']}). The strongest flags:\n" + "\n".join(lines) +
                        f"\n{r['status']}")
        elif re.search(r"importan|driver|feature|explain why|what matters", q):
            r = run("feature_importance", horizon=h, top=5)
            if "error" in r:
                text = f"I could not retrieve feature importance: {r['error']}"
            else:
                groups = ", ".join(f"{k} {v}" for k, v in r["group_gain_share"].items())
                feats = ", ".join(f"{x['feature']} ({x['gain_share']})" for x in r["top_features"])
                text = (f"For the {h} hour horizon the LightGBM weather model's gain shares by input group are: {groups}. "
                        f"The largest single features are {feats}. {r['caveat']}")
        elif re.search(r"calibrat|coverage|interval|uncertain", q) and not dates:
            r = run("calibration", model="lightgbm_weather", horizon=h)
            if "error" in r:
                text = f"I could not retrieve calibration results: {r['error']}"
            else:
                lines = [f"- {m['method']}: 90 percent interval covered {m['coverage_of_90_interval']} of outcomes, mean width "
                         f"{m['mean_width_90_mw']} MW, WIS {m['wis_mw']} MW" for m in r["methods"]]
                text = (f"Interval calibration of lightgbm_weather at the {h} hour horizon (nominal coverage 0.9):\n" + "\n".join(lines))
        elif re.search(r"compar|best|rank|accura|which model|outperform|better", q):
            by = "season" if "season" in q else "load_regime" if re.search(r"regime|peak|high load", q) else "overall"
            r = run("compare_models", horizon=h, metric="mae", by=by, include_oracle=False)
            if "error" in r:
                text = f"I could not compare models: {r['error']}"
            else:
                rows = r["ranking"][:6] if by == "overall" else r["ranking"]
                if by != "overall":
                    best = {}
                    for x in rows:
                        best.setdefault(x[by], x)
                    lines = [f"- {k}: {x['model']} ({x['regime']} weather), MAE {x['mae']} MW" for k, x in best.items()]
                    text = f"Lowest test MAE at the {h} hour horizon by {by.replace('_', ' ')}:\n" + "\n".join(lines) + f"\n{r['note']}"
                else:
                    lines = [f"- {x['model']} ({x['family']}, {x['regime']} weather): MAE {x['mae']} MW" for x in rows]
                    text = f"Test-period ranking at the {h} hour horizon by MAE:\n" + "\n".join(lines) + f"\n{r['note']}"
        elif re.search(r"forecast|predict", q) and dates:
            t = dates[0] if len(dates[0]) > 10 else dates[0] + " 22:00"
            r = run("get_forecast", target_time=t, horizon=h, model=None)
            if "error" in r:
                text = f"I could not retrieve that forecast: {r['error']}"
            else:
                text = (f"{r['model']} ({r['weather_information']} weather), issued {r['issued_at_utc']}, forecast {r['forecast_mw']} MW "
                        f"for the hour ending {r['target_hour_ending_utc']}, with a 90 percent interval from {r['interval_90_low_mw']} "
                        f"to {r['interval_90_high_mw']} MW ({r['interval_method']}).")
                if r.get("actual_mw") is not None:
                    inside = "inside" if r["actual_inside_90_interval"] else "outside"
                    text += f" Actual demand was {r['actual_mw']} MW, an error of {r['error_mw']} MW, {inside} the interval."
                text += " " + r["note"]
        elif dates:
            end = dates[1] if len(dates) > 1 else str(pd.Timestamp(dates[0]) + pd.Timedelta(days=1))
            r = run("query_history", start=dates[0], end=end, aggregate="summary")
            if "error" in r:
                text = f"I could not retrieve demand history: {r['error']}"
            else:
                text = (f"Between {r['start']} and {r['end']} observed demand averaged {r['mean_mw']} MW, with a minimum of {r['min_mw']} MW "
                        f"and a maximum of {r['max_mw']} MW at {r['peak_time_local']} local time. Mean temperature across the weather "
                        f"points was {r['mean_temperature_c']} C.")
        else:
            r = run("list_models")
            names = ", ".join(sorted({m["model"] for m in r.get("models", [])}))
            text = ("I can retrieve stored forecasts, summarise observed demand, compare models, show interval calibration, "
                    f"describe feature importance and list unusual observations. Models with results: {names}.")
        return _finish(question, text, self.name, calls)


class LLMBackend:
    """Claude with tool use. Requires the `anthropic` package and credentials."""

    name = "llm"

    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        import anthropic

        self.model = model
        self.client = client or anthropic.Anthropic()

    def ask(self, question: str) -> Answer:
        messages = [{"role": "user", "content": question}]
        schemas = [{**t, "strict": True} for t in tools.TOOL_SCHEMAS]
        calls: list[dict] = []
        response = None
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.messages.create(
                model=self.model, max_tokens=16000, system=SYSTEM, tools=schemas, messages=messages,
                output_config={"effort": "medium"},
            )
            if response.stop_reason == "refusal":
                return _finish(question, "The model declined to answer this request.", self.name, calls, refused=True)
            if response.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                res = tools.call(block.name, dict(block.input))
                calls.append({"tool": block.name, "args": dict(block.input), "result": res})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": verify.dumps(res),
                                "is_error": "error" in res})
            messages.append({"role": "user", "content": results})
        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "tool_use":
            text = "I could not finish within the allowed number of tool calls."
        return _finish(question, text, self.name, calls)


def credentials_available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return os.path.isdir(os.path.expanduser("~/.config/anthropic"))


def make_backend(prefer: str = "auto"):
    """prefer: auto | llm | offline. `auto` uses Claude when credentials exist, else the offline router."""
    if prefer == "offline" or (prefer == "auto" and not credentials_available()):
        return OfflineBackend()
    try:
        return LLMBackend()
    except Exception:
        if prefer == "llm":
            raise
        return OfflineBackend()


def ask(question: str, prefer: str = "auto") -> Answer:
    return make_backend(prefer).ask(question)

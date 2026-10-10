"""Streamlit dashboard over the stored study results.

    streamlit run app/dashboard.py

Read-only. All numbers come from `gridcast.copilot.tools` and the result
tables, the same sources the API and the copilot use.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from gridcast import config
from gridcast.copilot import agent, tools

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

st.set_page_config(page_title="ERCOT demand forecasting study", layout="wide")
st.title("ERCOT electricity demand forecasting study")
st.caption("Stored out-of-sample backtest results. Timestamps are UTC, hour ending. This is a research dashboard, "
           "not an operational tool.")


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    return pd.read_parquet(config.RESULTS / "serving" / "forecasts.parquet")


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.TABLES / f"{name}.csv")


def need(path) -> bool:
    if not path.exists():
        st.warning(f"`{path.relative_to(config.ROOT)}` has not been built yet. Run the pipeline in the README first.")
        return False
    return True


tab_fc, tab_cmp, tab_unc, tab_anom, tab_copilot = st.tabs(
    ["Forecast explorer", "Model comparison", "Uncertainty", "Unusual observations", "Grid Copilot"])

with tab_fc:
    if need(config.RESULTS / "serving" / "forecasts.parquet"):
        f = load_forecasts()
        c1, c2, c3, c4 = st.columns(4)
        model = c1.selectbox("Model", sorted(f["model"].unique()), index=sorted(f["model"].unique()).index("lightgbm_weather")
                             if "lightgbm_weather" in set(f["model"]) else 0)
        h = c2.selectbox("Horizon (hours ahead)", sorted(int(x) for x in f["h"].unique()), index=1)
        lo, hi = f["target_time"].min().date(), f["target_time"].max().date()
        start = c3.date_input("Week starting", value=pd.Timestamp("2024-07-05").date(), min_value=lo, max_value=hi)
        days = c4.slider("Days shown", 2, 21, 7)
        a = pd.Timestamp(start, tz="UTC")
        g = f[(f["model"] == model) & (f["h"] == h) & (f["target_time"] >= a) & (f["target_time"] < a + pd.Timedelta(days=days))]
        if g.empty:
            st.info("No stored forecasts in that window.")
        else:
            g = g.assign(t=g["target_time"].dt.tz_convert(None))
            ok = g.dropna(subset=["y"])
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("MAE in window", f"{(ok.y - ok.yhat).abs().mean():,.0f} MW")
            k2.metric("MAPE in window", f"{(100 * (ok.y - ok.yhat).abs() / ok.y).mean():.2f} %")
            k3.metric("90% interval coverage", f"{((ok.y >= ok['q0.05']) & (ok.y <= ok['q0.95'])).mean():.0%}")
            k4.metric("Mean 90% interval width", f"{(g['q0.95'] - g['q0.05']).mean():,.0f} MW")
            x = alt.X("t:T", title="Target hour (UTC, hour ending)")
            band = alt.Chart(g).mark_area(opacity=0.18, color=BLUE).encode(
                x=x, y=alt.Y("q0.05:Q", title="Demand (MW)", scale=alt.Scale(zero=False)), y2="q0.95:Q")
            long = g.melt(id_vars=["t"], value_vars=["y", "yhat"], var_name="series", value_name="mw")
            long["series"] = long["series"].map({"y": "Actual", "yhat": f"Forecast ({h} h ahead)"})
            lines = alt.Chart(long).mark_line(strokeWidth=2).encode(
                x=x, y="mw:Q",
                color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", f"Forecast ({h} h ahead)"], range=[INK, BLUE]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("t:T", title="Hour ending (UTC)", format="%Y-%m-%d %H:%M"), "series:N",
                         alt.Tooltip("mw:Q", title="MW", format=",.0f")])
            st.altair_chart((band + lines).properties(height=380), use_container_width=True)
            st.caption(f"Shaded band: 90% prediction interval ({g['method'].iloc[0].replace('_', ' ')}). "
                       f"Weather information used: {g['regime'].iloc[0]}.")
            with st.expander("Table view"):
                st.dataframe(g[["origin", "target_time", "yhat", "q0.05", "q0.95", "y"]].rename(
                    columns={"yhat": "forecast_mw", "y": "actual_mw", "q0.05": "low_90", "q0.95": "high_90"}), hide_index=True)

with tab_cmp:
    if need(config.TABLES / "benchmark_overall.csv"):
        c1, c2, c3 = st.columns(3)
        h = c1.selectbox("Horizon", config.HEADLINE_HORIZONS, index=1, key="cmp_h")
        metric = c2.selectbox("Metric", ["mae", "rmse", "mape", "mase", "bias"])
        by = c3.selectbox("Split", ["overall", "season", "load_regime"])
        oracle = st.checkbox("Also show oracle-weather results (not operational: uses observed future weather)")
        r = tools.call("compare_models", {"horizon": h, "metric": metric, "by": by, "include_oracle": oracle})
        if "error" in r:
            st.error(r["error"])
        else:
            d = pd.DataFrame(r["ranking"])
            d["label"] = d["model"] + " · " + d["regime"]
            if by == "overall":
                ch = alt.Chart(d).mark_bar(cornerRadiusEnd=4, height=14).encode(
                    y=alt.Y("label:N", sort=alt.EncodingSortField(metric, order="ascending"), title=None),
                    x=alt.X(f"{metric}:Q", title=f"{metric.upper()} ({r['unit']})"),
                    color=alt.Color("family:N", scale=alt.Scale(domain=["statistical", "ml", "deep"], range=[BLUE, ORANGE, "#1baf7a"]),
                                    legend=alt.Legend(title="Family", orient="top")),
                    tooltip=["model", "family", "regime", alt.Tooltip(f"{metric}:Q", format=",.2f"), "n"])
                st.altair_chart(ch.properties(height=28 * len(d) + 40), use_container_width=True)
            else:
                st.dataframe(d.pivot_table(index=["model", "regime"], columns=by, values=metric).round(2))
            st.caption(r["note"])
            with st.expander("Table view"):
                st.dataframe(d.drop(columns="label"), hide_index=True)

with tab_unc:
    if need(config.TABLES / "probabilistic_overall.csv"):
        p = table("probabilistic_overall")
        c1, c2 = st.columns(2)
        model = c1.selectbox("Model", sorted(p["model"].unique()), key="unc_m")
        h = c2.selectbox("Horizon", sorted(int(x) for x in p["h"].unique()), index=1, key="unc_h")
        g = p[(p["model"] == model) & (p["h"] == h)]
        show = g[["method", "cov50", "cov80", "cov90", "cov95", "width90", "wis", "pinball"]].rename(columns={
            "cov50": "coverage 50%", "cov80": "coverage 80%", "cov90": "coverage 90%", "cov95": "coverage 95%",
            "width90": "mean width 90% (MW)", "wis": "WIS (MW)", "pinball": "pinball (MW)"})
        st.dataframe(show.round(3), hide_index=True)
        st.caption("Coverage is the share of test hours whose actual demand fell inside the interval. A calibrated 90% "
                   "interval covers close to 0.90. Narrower is better only at equal coverage.")
        rel = table("probabilistic_reliability")
        rg = rel[(rel["model"] == model) & (rel["h"] == h)]
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=MUTED, strokeDash=[4, 4], strokeWidth=1).encode(x="x", y="y")
        ch = alt.Chart(rg).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
            x=alt.X("nominal:Q", title="Nominal quantile level"), y=alt.Y("empirical:Q", title="Share of actuals at or below the quantile"),
            color=alt.Color("method:N", legend=alt.Legend(title=None, orient="top")), tooltip=["method", "nominal", alt.Tooltip("empirical:Q", format=".3f")])
        st.altair_chart((diag + ch).properties(height=360), use_container_width=True)

with tab_anom:
    st.caption("Statistical flags from a seasonal-residual detector. Flags are unverified: a flag is not evidence of a "
               "grid event or a data error. Detector accuracy is only measured on simulated anomalies (see the README).")
    c1, c2 = st.columns(2)
    a = c1.date_input("From", value=pd.Timestamp("2024-07-01").date(), key="an_a")
    b = c2.date_input("To", value=pd.Timestamp("2024-07-31").date(), key="an_b")
    r = tools.call("unusual_observations", {"start": str(a), "end": str(b), "top": 25})
    if "error" in r:
        st.error(r["error"])
    else:
        st.write(f"{r['flagged_hours_in_window']} of {r['hours_in_window']} hours exceed the score threshold of {r['threshold_score']}.")
        st.dataframe(pd.DataFrame(r["top"]), hide_index=True)

with tab_copilot:
    st.caption("The copilot only reads stored results through Python tools. Every number in an answer is checked against "
               "the tool outputs. It cannot operate grid equipment.")
    backend = "llm" if agent.credentials_available() else "offline"
    st.write(f"Backend: **{backend}**" + ("" if backend == "llm" else " (no Anthropic credentials found; using the deterministic router)"))
    q = st.text_input("Question", "Which model is most accurate 24 hours ahead?")
    if st.button("Ask") and q.strip():
        try:
            ans = agent.ask(q)
            st.markdown(ans.text)
            (st.success if ans.grounded else st.error)(
                f"Numeric grounding: {'all' if ans.grounded else 'NOT all'} of {ans.numbers_checked} numbers are supported by tool outputs"
                + ("" if ans.grounded else f". Unsupported: {ans.unsupported_numbers}"))
            with st.expander("Tool calls (evidence)"):
                st.json(ans.tool_calls)
        except Exception as e:
            st.error(f"Copilot unavailable: {type(e).__name__}: {e}")

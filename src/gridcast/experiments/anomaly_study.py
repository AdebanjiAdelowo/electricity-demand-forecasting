"""Anomaly analysis in two clearly separated parts.

Part A (SIMULATED): plant anomalies of known type and size into the first test
year and measure each detector's sensitivity. Thresholds are set on the clean
series so that every detector raises alarms in the same fraction of hours.

Part B (REAL, UNLABELLED): score the real series and check whether a short list
of publicly documented events falls inside flagged periods. This is a
qualitative consistency check. No detection accuracy is claimed for real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from gridcast import config, tracking
from gridcast.anomaly import detectors as det
from gridcast.anomaly.inject import TYPES, inject
from gridcast.data import dataset

MAGNITUDES = [0.02, 0.05, 0.10, 0.20]
ALARM_RATE = 0.005  # fraction of clean hours each detector is allowed to flag
TOLERANCE_H = 2

# Publicly documented events used only for the qualitative check in part B.
DOCUMENTED_EVENTS = [
    {"event": "Hurricane Harvey landfall and Houston flooding", "start": "2017-08-25", "end": "2017-08-31",
     "source": "NOAA NHC Tropical Cyclone Report AL092017"},
    {"event": "Winter Storm Uri, ERCOT rotating outages", "start": "2021-02-14", "end": "2021-02-20",
     "source": "FERC/NERC joint report on the February 2021 cold weather outages"},
    {"event": "Winter Storm Elliott cold snap", "start": "2022-12-22", "end": "2022-12-26",
     "source": "FERC/NERC joint report on Winter Storm Elliott"},
    {"event": "Hurricane Beryl landfall, Houston-area outages", "start": "2024-07-08", "end": "2024-07-12",
     "source": "NOAA NHC Tropical Cyclone Report AL022024"},
]


def build_detectors(df: pd.DataFrame, fold):
    """Fit the monitoring forecasters and the isolation forest on clean history before the fold."""
    m1 = det.fit_residual_model(df, fold, h=1, weather=False)
    m24 = det.fit_residual_model(df, fold, h=24, weather=True)
    fr1 = det.ForecastResidual(m1, 1, "none")
    fr24 = det.ForecastResidual(m24, 24, "oracle")
    hist = df[(df.index >= fold.cal_start - pd.Timedelta(days=3 * 365)) & (df.index < fold.cal_start)]
    iso = det.IsoForest().fit(det.detector_features(hist, fr1.residual(hist)))
    return fr1, fr24, iso


def score_all(df: pd.DataFrame, fr1, fr24, iso) -> pd.DataFrame:
    r1 = fr1.residual(df)
    z24 = det.robust_z(fr24.residual(df))
    return pd.DataFrame({
        "robust_seasonal": det.score_robust_seasonal(df),
        "forecast_residual_h1": det.robust_z(r1).abs(),
        "forecast_residual_h24_weather": z24.abs(),
        "cusum_h24_weather": det.cusum(z24),
        "isolation_forest": iso.score(det.detector_features(df, r1)),
        "flatline": det.flatline_score(df["demand"]),
    })


def event_recall(alarm: np.ndarray, events: pd.DataFrame, tol: int = TOLERANCE_H) -> tuple[float, float]:
    hits, delays = 0, []
    for e in events.itertuples():
        seg = alarm[e.start_pos: e.end_pos + 1 + tol]
        if seg.any():
            hits += 1
            delays.append(int(np.argmax(seg)))
    return hits / max(len(events), 1), float(np.mean(delays)) if delays else float("nan")


def simulated(df: pd.DataFrame, fold, n_events: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    fr1, fr24, iso = build_detectors(df, fold)
    lo = fold.test_start - pd.Timedelta(days=60)
    hi = fold.test_end
    window = df[(df.index >= lo - pd.Timedelta(days=60)) & (df.index < hi)]
    ev = (window.index >= fold.test_start) & (window.index < hi)
    clean = score_all(window, fr1, fr24, iso)
    thr = clean[ev].quantile(1 - ALARM_RATE)
    thr["flatline"] = 4.5  # five identical consecutive hourly values
    rows = [{"detector": d, "threshold": float(thr[d]), "clean_alarm_rate": float((clean.loc[ev, d] > thr[d]).mean())} for d in clean]
    thresholds = pd.DataFrame(rows)
    out = []
    for kind in TYPES:
        for mag in ([0.0] if kind == "flatline" else MAGNITUDES):
            inj = inject(window, fold.test_start, hi - pd.Timedelta(days=8), kind, mag, n_events, seed)
            sc = score_all(inj.df, fr1, fr24, iso)
            y = inj.labels[ev].to_numpy()
            off = int(np.flatnonzero(ev)[0])
            events = inj.events.assign(start_pos=inj.events.start_pos - off, end_pos=inj.events.end_pos - off)
            for d in sc:
                s = sc.loc[ev, d].fillna(0).to_numpy()
                alarm = s > thr[d]
                rec, delay = event_recall(alarm, events)
                out.append({"type": kind, "magnitude": mag, "detector": d, "n_events": len(events),
                            "anomalous_hours": int(y.sum()), "event_recall": rec, "mean_delay_h": delay,
                            "false_alarm_rate": float(alarm[~y].mean()),
                            "avg_precision": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
                            "base_rate": float(y.mean())})
    return pd.DataFrame(out), thresholds


def real_events(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Qualitative check on real data with the label-free seasonal detector over the full history."""
    score = det.score_robust_seasonal(df)
    resid = det.seasonal_residual(df["demand"])
    thr = float(score.quantile(1 - ALARM_RATE))
    alarm = score > thr
    rows = []
    for e in DOCUMENTED_EVENTS:
        m = (df.index >= pd.Timestamp(e["start"], tz=config.LOCAL_TZ)) & (df.index < pd.Timestamp(e["end"], tz=config.LOCAL_TZ))
        rows.append({**e, "hours": int(m.sum()), "flagged_hours": int(alarm[m].sum()), "flagged": bool(alarm[m].any()),
                     "max_score": float(score[m].max()), "largest_relative_deviation": float(resid[m].loc[resid[m].abs().idxmax()]),
                     "expected_flagged_hours_by_chance": round(float(m.sum() * ALARM_RATE), 2)})
    # the most extreme flagged episodes, for manual inspection (not verified events)
    grp = (alarm != alarm.shift()).cumsum()[alarm]
    ep = pd.DataFrame({"score": score[alarm], "resid": resid[alarm], "g": grp})
    top = ep.groupby("g").agg(start=("score", lambda s: s.index[0]), end=("score", lambda s: s.index[-1]),
                              hours=("score", "size"), max_score=("score", "max"),
                              largest_relative_deviation=("resid", lambda s: s.loc[s.abs().idxmax()]))
    top = top.sort_values("max_score", ascending=False).head(15).reset_index(drop=True)
    top["start_local"] = top["start"].dt.tz_convert(config.LOCAL_TZ).astype(str)
    top["status"] = "unverified flag"
    return pd.DataFrame(rows), top.drop(columns=["start", "end"])


def main(argv=None):
    config.ensure_dirs()
    df = dataset.load()
    fold = config.make_folds(config.TEST_START, config.TEST_START + pd.DateOffset(years=1), months=12)[0]
    with tracking.Run("anomaly_study", {"alarm_rate": ALARM_RATE, "magnitudes": MAGNITUDES, "types": TYPES}, [fold]) as run:
        sim, thr = simulated(df, fold)
        sim.to_csv(config.TABLES / "anomaly_simulated.csv", index=False)
        thr.to_csv(config.TABLES / "anomaly_thresholds.csv", index=False)
        ev, top = real_events(df)
        ev.to_csv(config.TABLES / "anomaly_real_events.csv", index=False)
        top.to_csv(config.TABLES / "anomaly_top_flags.csv", index=False)
        run.log(mean_recall=sim.groupby("detector")["event_recall"].mean().round(3).to_dict())
    pd.set_option("display.width", 220)
    print(sim.pivot_table(index=["type", "magnitude"], columns="detector", values="event_recall").round(2))
    print(sim.pivot_table(index=["type", "magnitude"], columns="detector", values="avg_precision").round(2))
    print(thr.round(3))
    print(ev[["event", "hours", "flagged_hours", "max_score", "largest_relative_deviation", "expected_flagged_hours_by_chance"]])
    print(top)


if __name__ == "__main__":
    main()

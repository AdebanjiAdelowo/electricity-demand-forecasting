"""Command line entry point: `gridcast <stage>`.

Stages run in the order the study was built. Each one is restartable.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="gridcast", description=__doc__)
    ap.add_argument("stage", choices=["data", "benchmark", "report", "probabilistic", "anomaly", "study", "serving",
                                      "figures", "copilot-eval", "ask"])
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.stage == "data":
        from gridcast.data import dataset

        df = dataset.build(download="--no-download" not in a.rest)
        print(f"wrote {dataset.PROCESSED} ({len(df)} rows)")
    elif a.stage == "benchmark":
        from gridcast.experiments import benchmark

        benchmark.main(a.rest)
    elif a.stage == "report":
        from gridcast.experiments import report

        report.main(a.rest)
    elif a.stage == "probabilistic":
        from gridcast.experiments import probabilistic

        probabilistic.main(a.rest)
    elif a.stage == "anomaly":
        from gridcast.experiments import anomaly_study

        anomaly_study.main(a.rest)
    elif a.stage == "study":
        from gridcast.experiments import studies

        studies.main(a.rest)
    elif a.stage == "serving":
        from gridcast.experiments import serving

        serving.main(a.rest)
    elif a.stage == "figures":
        from gridcast.viz import plots

        plots.main(a.rest)
    elif a.stage == "copilot-eval":
        from gridcast.experiments import copilot_eval

        copilot_eval.main(a.rest)
    elif a.stage == "ask":
        from gridcast.copilot import agent

        ans = agent.ask(" ".join(a.rest))
        print(ans.text)
        print(f"\n[backend={ans.backend} grounded={ans.grounded} numbers_checked={ans.numbers_checked} "
              f"tools={[c['tool'] for c in ans.tool_calls]}]", file=sys.stderr)


if __name__ == "__main__":
    main()

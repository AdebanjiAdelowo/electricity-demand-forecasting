import importlib.util
import sys

import pandas as pd

from gridcast import config
from gridcast.experiments import copilot_eval, studies
from gridcast.viz import plots


def test_kaggle_default_is_plan_only(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("gridcast_gpu_plan", config.ROOT / "kaggle/run_gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["run_gpu.py", "--src", "/not/a/real/dataset"])
    monkeypatch.setattr(
        module, "sh", lambda *_: (_ for _ in ()).throw(AssertionError("No execution authorized"))
    )
    module.main()
    assert "PENDING: deep_benchmark" in capsys.readouterr().out


def test_architecture_studies_dispatch_without_training(monkeypatch):
    calls = []
    monkeypatch.setattr(studies, "_deep_study", lambda *args: calls.append(args))
    studies.lookback("gru")
    studies.size("tcn")
    studies.seeds("nbeats", n=2)
    assert calls[0][0] == "lookback_gru" and calls[0][2] == "gru"
    assert calls[1][0] == "size_tcn" and calls[1][2] == "tcn"
    assert list(calls[2][1]) == ["seed 0", "seed 1"]


def test_legacy_figures_do_not_label_unverified_weather_operational():
    assert plots.OPERATIONAL == ["none"]
    table = pd.DataFrame({"regime": ["none", "forecast", "oracle"], "mae": [1, 2, 3]})
    assert plots._op(table).regime.tolist() == ["none"]


def test_copilot_offline_evaluation_uses_expected_tool_and_values(monkeypatch):
    monkeypatch.setattr(
        copilot_eval,
        "_cases",
        lambda: [
            {
                "q": "Which model is most accurate 24 hours ahead?",
                "tool": "compare_models",
                "expect": ["ridge", "1234.5"],
                "kind": "comparison",
            }
        ],
    )
    monkeypatch.setattr(
        copilot_eval.tools,
        "call",
        lambda *_: {
            "ranking": [{"model": "ridge", "family": "ml", "regime": "none", "mae": 1234.5}],
            "note": "Fixture comparison",
        },
    )
    result = copilot_eval.evaluate(copilot_eval.agent.OfflineBackend())
    assert result.passed.all() and result.grounded.all()

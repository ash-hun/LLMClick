"""`evaluation_compare` evaluation/compare_evaluation.py: config rules, the primary metric, the verdict against the base, and the chart."""

import pytest

from evaluation.compare_evaluation import CompareConfig, polygon_chart, primary, verdict


def test_config_rules() -> None:
    runs = [{"label": "base", "experiment": "e", "checkpoint": "base"}, {"label": "final", "experiment": "e"}]
    tasks = {"tasks": [{"name": "hellaswag"}]}
    assert CompareConfig(recipe="evaluation_compare", name="c", runs=runs, benchmarks=tasks).base.label == "base"
    for broken, message in [({"runs": runs[:1]}, "at least 2"), ({"runs": [runs[0], runs[0]]}, "labels must differ"),
                            ({"runs": [runs[1], {**runs[1], "label": "other"}]}, "checkpoint: base"),
                            ({"runs": runs, "benchmarks": None}, "benchmarks, rows, decision or embedding"),
                            ({"runs": runs, "axes": {"a": ["gsm8k"]}}, "no benchmark, rows:")]:
        with pytest.raises(ValueError, match=message):
            CompareConfig(recipe="evaluation_compare", name="c", **{"benchmarks": tasks, **broken})


def test_primary_metric_verdict_and_chart() -> None:
    assert primary({"acc": 0.5, "acc_norm": 0.6, "acc_stderr": 0.1}) == "acc_norm" and primary({"bleu": 1.0}) == "bleu"
    assert primary({}) is None
    assert verdict(0.6, 0.5, 0.02, 0.02) == {"delta": pytest.approx(0.1), "margin": pytest.approx(0.0554, abs=1e-3), "decided": True}
    assert verdict(0.52, 0.5, 0.02, 0.02)["decided"] is False and verdict(0.6, 0.5, None, 0.1) == {"delta": pytest.approx(0.1), "decided": None}
    svg = polygon_chart({"knowledge": {"base": 0.3, "final": 0.6}, "korean": {"base": 0.2, "final": 0.4}}, ["base", "final"])
    assert svg.count("<polygon") == 4 + 2 and "knowledge" in svg and ">final<" in svg  # four rings, two runs

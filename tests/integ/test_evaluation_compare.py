"""`evaluation_compare` end to end: several checkpoints on the same benchmarks and rows, sharing stages with the single-model recipes."""

import json
from typing import Any
from pathlib import Path

import pytest

from tests.recipe_inputs import decision_raw, raw
from tests.recipe_fakes import FakeHarness
from core.progress import StateProgress
from core import pipeline


def config(experiment: str, out: Path, **keys: Any) -> dict[str, Any]:
    return {"pipeline": {"recipe": "evaluation_compare", "name": "cmp", "output_dir": str(out), "device": "cpu"},
            "runs": [{"label": "base", "experiment": experiment, "checkpoint": "base"},
                     {"label": "final", "experiment": experiment, "checkpoint": "validate"}],
            "evaluation": {"batch_size": 8}, **keys}


def test_compares_runs_on_benchmarks_and_rows_and_shares_their_stages(trained: dict[str, Any], tmp_path: Path, fake: FakeHarness) -> None:
    held_out = trained["stages"]["data"]["validation"]
    single = {"pipeline": {"recipe": "evaluation_benchmark", "name": "one", "output_dir": str(tmp_path), "device": "cpu"},
              "model": {"experiment": trained["directory"], "checkpoint": "validate"},
              "benchmarks": {"tasks": [{"name": "hellaswag"}], "limit": 4}, "evaluation": {"batch_size": 8}}
    pipeline.build(single).run()  # the final checkpoint on hellaswag, measured before any comparison asked for it
    custom = {"pipeline": {"recipe": "evaluation_custom", "name": "rows", "output_dir": str(tmp_path), "device": "cpu"},
              "model": {"experiment": trained["directory"], "checkpoint": "validate"},
              "data": {"sources": [{"name": "local_jsonl", "path": held_out}]}, "evaluation": {"batch_size": 8}}
    custom_result = pipeline.build(custom).run()

    progress = StateProgress()
    body = config(trained["directory"], tmp_path, benchmarks={"tasks": [{"name": "hellaswag"}, {"name": "gsm8k"}], "limit": 4},
                  rows={"sources": [{"name": "local_jsonl", "path": held_out}]}, axes={"knowledge": ["hellaswag", "gsm8k"]})
    built = pipeline.build(body, progress)
    assert built.plan() == ["score:base:hellaswag", "score:base:gsm8k", "score:final:hellaswag", "score:final:gsm8k",
                            "rows", "rows:base", "rows:final", "report"]
    result = built.run()
    states = progress.snapshot()["stages"]
    assert states["score:final:hellaswag"] == "cached" and states["rows"] == "cached" and states["rows:final"] == "cached"
    assert states["score:base:hellaswag"] == "done" and states["rows:base"] == "done" and states["report"] == "done"
    assert [call["tasks"] for call in fake.calls] == [["hellaswag"], ["hellaswag"], ["gsm8k"], ["gsm8k"]]  # 1 single + 3 new

    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    assert report["base"] == "base" and [run["label"] for run in report["runs"]] == ["base", "final"]
    final = report["benchmarks"]["hellaswag"]["final"]
    assert final["metric"] == "acc" and final["value"] == 0.25 and final["stderr"] == 0.05 and final["n"] == 4  # the cached single run
    assert report["benchmarks"]["hellaswag"]["base"]["value"] == 0.5
    assert report["comparison"]["hellaswag"]["final"] == {"delta": -0.25, "margin": pytest.approx(0.1386, abs=1e-3), "decided": True}
    assert report["rows"]["loss"]["final"] == pytest.approx(custom_result["stages"]["report"]["scores"]["loss"])
    assert report["comparison"]["rows:loss"]["final"]["decided"] is None
    assert report["axes"] == {"knowledge": {"base": pytest.approx((0.5 + 0.75) / 2), "final": pytest.approx((0.25 + 1.0) / 2)}}
    assert Path(report["chart"]).read_text().count("<polygon") == 6
    summary = Path(report["summary"]).read_text()
    assert "| hellaswag | acc | 0.5000 ± 0.0500 | 0.2500 ± 0.0500 (-0.2500) |" in summary
    assert "| loss |" in summary and "?)" in summary and "![axes](axes.svg)" in summary

    again = pipeline.build(body, StateProgress()).run()
    assert json.dumps(again) == json.dumps(result)


def test_decision_and_embedding_sections_join_the_comparison(tmp_path: Path, tiny_model: Path) -> None:
    decision = pipeline.build(decision_raw("llm_decision_sft", tiny_model, tmp_path, "readout")).run()
    body = {"pipeline": {"recipe": "evaluation_compare", "name": "cmp-dec", "output_dir": str(tmp_path), "device": "cpu"},
            "runs": [{"label": "base", "experiment": decision["directory"], "checkpoint": "base"},
                     {"label": "final", "experiment": decision["directory"], "checkpoint": "validate"}],
            "decision": {"data": {"sources": [{"name": "local_jsonl", "path": decision["stages"]["data"]["validation"]}]}, "think": "off"},
            "axes": {"decide": ["decision:nothink:accuracy"]}, "evaluation": {"batch_size": 4}}
    built = pipeline.build(body)
    assert built.plan() == ["questions", "decision:base", "decision:final", "report"]
    report = json.loads(Path(built.run()["stages"]["report"]["report"]).read_text())
    assert set(report["decision"]["nothink"]) >= {"accuracy", "nll", "ece", "questions"}
    assert report["decision"]["nothink"]["accuracy"]["final"] == pytest.approx(decision["stages"]["validate"]["metrics"]["accuracy"])
    assert report["comparison"]["decision:nothink:accuracy"]["final"]["decided"] is None
    assert report["axes"]["decide"]["final"] == pytest.approx(report["decision"]["nothink"]["accuracy"]["final"])
    assert report["contamination"]["base"] is None and report["contamination"]["final"]["items"] > 0
    assert "## Decision" in Path(report["summary"]).read_text()

    embedding = pipeline.build(raw("embedding_contrastive", tiny_model, tmp_path)).run()
    body = {"pipeline": {"recipe": "evaluation_compare", "name": "cmp-emb", "output_dir": str(tmp_path), "device": "cpu"},
            "runs": [{"label": "base", "experiment": embedding["directory"], "checkpoint": "base"},
                     {"label": "final", "experiment": embedding["directory"], "checkpoint": "validate"}],
            "embedding": {"tasks": ["MockSTSTask"]}, "axes": {"sts": ["embedding:MockSTSTask"]}, "evaluation": {"batch_size": 4}}
    built = pipeline.build(body)
    assert built.plan() == ["embedding:base:MockSTSTask", "embedding:final:MockSTSTask", "report"]
    report = json.loads(Path(built.run()["stages"]["report"]["report"]).read_text())
    assert set(report["embedding"]["MockSTSTask"]) == {"base", "final"} and "embedding:MockSTSTask" in report["comparison"]
    assert report["contamination"] == {"base": None, "final": None}  # MTEB tasks are not rows of the experiment


def test_a_hub_model_serves_as_the_baseline(trained: dict[str, Any], tmp_path: Path, fake: FakeHarness, tiny_model: Path) -> None:
    body = {"pipeline": {"recipe": "evaluation_compare", "name": "cmp-hub", "output_dir": str(tmp_path), "device": "cpu"},
            "runs": [{"label": "qwen", "name": str(tiny_model)},
                     {"label": "final", "experiment": trained["directory"], "checkpoint": "validate"}],
            "benchmarks": {"tasks": [{"name": "hellaswag"}], "limit": 4}, "evaluation": {"batch_size": 2}}
    built = pipeline.build(body)
    assert built.config.base.label == "qwen"
    report = json.loads(Path(built.run()["stages"]["report"]["report"]).read_text())
    assert report["base"] == "qwen" and report["runs"][0]["name"] == str(tiny_model) and report["runs"][0]["checkpoint"] == "hub"
    assert "final" in report["comparison"]["hellaswag"] and report["contamination"] == {"qwen": None, "final": report["contamination"]["final"]}
    assert report["contamination"]["final"]["items"] == 1  # the fake harness logs one sample per task

"""`evaluation_benchmark`: one fingerprinted stage per benchmark over lm-evaluation-harness, and one report."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from core import pipeline
from core.progress import StateProgress
from evaluation.benchmark import PRESETS, Benchmarks, flat
from tests.evaluation.test_custom import trained  # noqa: F401 - the tiny trained experiment


def config(experiment: str, out: Path, **benchmarks: Any) -> dict[str, Any]:
    return {"pipeline": {"recipe": "evaluation_benchmark", "name": "bench", "output_dir": str(out), "device": "cpu"},
            "model": {"experiment": experiment, "checkpoint": "validate"},
            "benchmarks": {"limit": 4, **benchmarks}, "evaluation": {"batch_size": 2}}


class FakeHarness:
    """Stands in for lm_eval: records every call and answers with the shape `simple_evaluate` returns."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def simple_evaluate(self, **keys: Any) -> dict[str, Any]:
        self.calls.append(keys)
        (task,) = keys["tasks"]
        return {"results": {task: {"alias": task, "acc,none": 0.25 * len(self.calls), "acc_stderr,none": 0.05}},
                "n-samples": {task: {"original": 100, "effective": keys["limit"]}}, "n-shot": {task: keys["num_fewshot"] or 0},
                "versions": {task: 1}, "higher_is_better": {task: {"acc": True}},
                "samples": {task: [{"doc_id": 0, "acc": 1.0}]}, "config": {"model": keys["model"]}}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeHarness:
    import lm_eval

    harness = FakeHarness()
    monkeypatch.setattr(lm_eval, "simple_evaluate", harness.simple_evaluate)
    return harness


def test_presets_and_tasks_resolve_with_limits() -> None:
    tasks = Benchmarks(preset="korean", tasks=[{"name": "kmmlu", "num_fewshot": 0, "limit": 7}, {"name": "gsm8k"}], limit=3).resolved()
    assert [task.name for task in tasks] == ["haerae", "kobest", "click", "hrm8k", "kmmlu", "gsm8k"]
    assert {task.name: task.limit for task in tasks} == {"haerae": 3, "kobest": 3, "click": 3, "hrm8k": 3, "kmmlu": 7, "gsm8k": 3}
    assert next(task for task in tasks if task.name == "kmmlu").num_fewshot == 0  # replaced the preset entry
    with pytest.raises(ValueError, match="preset or at least one task"):
        Benchmarks()
    with pytest.raises(ValueError, match="choose from"):
        Benchmarks(preset="nope")
    assert flat({"acc,none": 0.5, "acc_stderr,none": 0.01, "alias": "x", "bleu,none": "N/A"}) == {"acc": 0.5, "acc_stderr": 0.01}


def test_every_preset_task_exists_in_the_harness() -> None:
    from lm_eval.tasks import TaskManager

    names = set(TaskManager().all_tasks)
    missing = [item["name"] for tasks in PRESETS.values() for item in tasks if item["name"] not in names]
    assert missing == []


def test_one_stage_per_benchmark_and_only_new_ones_are_scored(trained: dict[str, Any], tmp_path: Path, fake: FakeHarness) -> None:  # noqa: F811
    first = pipeline.build(config(trained["directory"], tmp_path, tasks=[{"name": "hellaswag", "num_fewshot": 2}, {"name": "gsm8k"}]))
    assert first.plan() == ["score:hellaswag", "score:gsm8k", "report"]
    result = first.run()
    assert [call["tasks"] for call in fake.calls] == [["hellaswag"], ["gsm8k"]]
    call = fake.calls[0]
    assert call["model"] == "hf" and call["model_args"]["pretrained"] == f"{trained['directory']}/validate/checkpoint"
    assert call["num_fewshot"] == 2 and call["limit"] == 4 and call["device"] == "cpu" and call["batch_size"] == 2
    assert call["gen_kwargs"] == {"max_gen_toks": 1024, "temperature": 0.0, "do_sample": False} and call["apply_chat_template"] is False
    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    assert report["benchmarks"]["hellaswag"]["scores"] == {"acc": 0.25, "acc_stderr": 0.05}
    assert report["benchmarks"]["gsm8k"] == {**report["benchmarks"]["gsm8k"], "n": {"gsm8k": 4}, "num_fewshot": 0, "limit": 4}
    assert report["settings"]["harness"]["name"] == "lm_eval" and report["settings"]["harness"]["version"]
    assert report["model"]["recipe"] == "llm_sft" and report["contamination"] is None
    assert Path(result["stages"]["score:gsm8k"]["samples"]).read_text().startswith('{"task": "gsm8k"')

    progress = StateProgress()
    more = pipeline.build(config(trained["directory"], tmp_path, tasks=[{"name": "hellaswag", "num_fewshot": 2}, {"name": "gsm8k"},
                                                                          {"name": "ifeval"}]), progress).run()
    assert [call["tasks"] for call in fake.calls] == [["hellaswag"], ["gsm8k"], ["ifeval"]]  # the first two came from cache
    assert progress.snapshot()["stages"] == {"score:hellaswag": "cached", "score:gsm8k": "cached", "score:ifeval": "done", "report": "done"}
    assert set(more["stages"]["report"]["scores"]) == {"hellaswag", "gsm8k", "ifeval"}
    changed = pipeline.build(config(trained["directory"], tmp_path, tasks=[{"name": "hellaswag", "num_fewshot": 5}]))
    assert changed.fingerprint("score:hellaswag") != first.fingerprint("score:hellaswag")  # few-shot is part of the identity


def test_base_checkpoint_scores_the_starting_weights(trained: dict[str, Any], tmp_path: Path, fake: FakeHarness) -> None:  # noqa: F811
    body = config(trained["directory"], tmp_path, tasks=[{"name": "hellaswag"}])
    body["model"]["checkpoint"] = "base"
    pipeline.build(body).run()
    trained_config = json.loads(json.dumps(__import__("yaml").safe_load(Path(trained["directory"], "config.yaml").read_text())))
    assert fake.calls[0]["model_args"]["pretrained"] == trained_config["model"]["name"]


def test_unknown_task_and_missing_harness_fail_clearly(trained: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    with pytest.raises(ValueError, match="unknown benchmark 'nope'"):
        pipeline.build(config(trained["directory"], tmp_path, tasks=[{"name": "nope"}])).run()
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    with pytest.raises(ImportError, match="uv sync --extra eval"):
        pipeline.build(config(trained["directory"], tmp_path, tasks=[{"name": "hellaswag"}])).run()


@pytest.mark.skipif(not os.environ.get("LLMCLICK_REAL_EVAL"), reason="downloads a benchmark; set LLMCLICK_REAL_EVAL=1")
def test_the_real_harness_scores_the_tiny_model(trained: dict[str, Any], tmp_path: Path) -> None:  # noqa: F811
    result = pipeline.build(config(trained["directory"], tmp_path, tasks=[{"name": "hellaswag", "num_fewshot": 0}])).run()
    scores = result["stages"]["report"]["scores"]["hellaswag"]
    assert 0.0 <= scores["acc"] <= 1.0 and "acc_stderr" in scores


def test_a_hub_model_is_scored_without_an_experiment(tiny_model: Path, tmp_path: Path, fake: FakeHarness) -> None:
    body = {"pipeline": {"recipe": "evaluation_benchmark", "name": "hub", "output_dir": str(tmp_path), "device": "cpu"},
            "model": {"name": str(tiny_model)}, "benchmarks": {"tasks": [{"name": "hellaswag"}], "limit": 4}, "evaluation": {"batch_size": 2}}
    result = pipeline.build(body).run()
    assert fake.calls[0]["model_args"] == {"pretrained": str(tiny_model), "dtype": "float32"}  # a directory: no revision
    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    assert report["model"]["name"] == str(tiny_model) and report["model"]["checkpoint"] == "hub" and report["model"]["experiment"] is None
    assert report["benchmarks"]["hellaswag"]["contamination"] is None  # nothing trained here
    hub = {**body, "model": {"name": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"}}
    assert pipeline.build(hub).fingerprint("score:hellaswag") != pipeline.build(body).fingerprint("score:hellaswag")
    with pytest.raises(ValueError, match="40-character commit"):
        pipeline.build({**body, "model": {"name": "Qwen/Qwen3.5-0.8B"}})
    with pytest.raises(ValueError, match="exactly one of"):
        pipeline.build({**body, "model": {"name": str(tiny_model), "experiment": "x"}})
    custom = {"pipeline": {"recipe": "evaluation_custom", "name": "c", "output_dir": str(tmp_path), "device": "cpu"},
              "model": {"name": str(tiny_model)}, "data": {"sources": [{"name": "local_jsonl", "path": "samples/llm_sft.jsonl"}]}}
    with pytest.raises(ValueError, match="only the benchmark and embedding recipes"):
        pipeline.build(custom).run()


def test_thinking_is_a_chat_template_argument(trained: dict[str, Any], tmp_path: Path, fake: FakeHarness) -> None:  # noqa: F811
    body = config(trained["directory"], tmp_path, tasks=[{"name": "gsm8k"}])
    with pytest.raises(ValueError, match="chat_template: true"):
        pipeline.build({**body, "thinking": True})
    pipeline.build({**body, "chat_template": True, "thinking": True}).run()
    assert fake.calls[-1]["apply_chat_template"] is True
    assert fake.calls[-1]["model_args"]["enable_thinking"] is True and fake.calls[-1]["model_args"]["think_end_token"] == "</think>"
    pipeline.build({**body, "chat_template": True, "thinking": False}).run()
    assert fake.calls[-1]["model_args"]["enable_thinking"] is False and "think_end_token" not in fake.calls[-1]["model_args"]
    plain = pipeline.build({**body, "chat_template": True})
    assert plain.fingerprint("score:gsm8k") != pipeline.build({**body, "chat_template": True, "thinking": False}).fingerprint("score:gsm8k")
    plain.run()
    assert "enable_thinking" not in fake.calls[-1]["model_args"]

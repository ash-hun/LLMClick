"""`llm_multi_teacher_on_policy_distillation` end to end on tiny models: per-domain teachers, the one-teacher case."""

import json
from typing import Any
from pathlib import Path

import pytest

from tests.recipe_inputs import raw
from core import pipeline_runner


def log(result: dict[str, Any]) -> list[dict[str, Any]]:
    lines = (Path(result["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_each_domain_is_measured_against_its_own_teacher(tiny_model: Path, tmp_path: Path) -> None:
    sft = pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path / "teacher", max_steps=20, lr=3e-3)).run()
    config = raw("llm_multi_teacher_on_policy_distillation", tiny_model, tmp_path)
    config["method"]["teachers"]["chat"]["name"] = sft["stages"]["train"]["checkpoint"]
    result = pipeline_runner.build(config).run()
    metrics = result["stages"]["validate"]["metrics"]
    assert {"reverse_kl", "reverse_kl/math", "reverse_kl/chat", "teacher_logp/chat", "reward"} <= set(metrics)
    assert abs(metrics["reverse_kl/math"]) < abs(metrics["reverse_kl/chat"])
    assert all("reward" in entry and "reverse_kl" in entry for entry in log(result))


def test_one_teacher_without_reward_trains_like_on_policy_distillation(tiny_model: Path, tmp_path: Path) -> None:
    rows = [json.loads(line) for line in Path("samples/llm_grpo.jsonl").read_text().splitlines()]
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps({**row, "domain": "math"}) + "\n" for row in rows))
    single = raw("llm_on_policy_distillation", tiny_model, tmp_path / "single")
    multi = raw("llm_multi_teacher_on_policy_distillation", tiny_model, tmp_path / "multi")
    multi["method"] = {"max_new_tokens": 3, "teachers": {"math": single["method"]["teacher"]}}
    for config in (single, multi):
        config["data"]["sources"][0]["path"] = str(data)
    expected = [entry["reverse_kl"] for entry in log(pipeline_runner.build(single).run())]
    assert [entry["reverse_kl"] for entry in log(pipeline_runner.build(multi).run())] == pytest.approx(expected, abs=1e-5)

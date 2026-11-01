"""`llm_sft` end to end on a tiny random model: three epochs over the sample rows halve the loss."""

import json
from pathlib import Path

from tests.recipe_inputs import raw
from core import pipeline_runner


def test_sft_learns_the_sample_rows(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_sft", tiny_model, tmp_path, max_steps=None, epochs=3, lr=3e-3)
    result = pipeline_runner.build(config).run()
    log = [json.loads(line) for line in (Path(result["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert [event["step"] for event in log] == list(range(1, 49)) and log[-1]["loss"] < log[0]["loss"] / 2


def test_tool_conversations_train_and_report_tool_call_exact(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_sft", tiny_model, tmp_path, max_length=128)
    config["data"]["sources"][0]["path"] = "samples/llm_tool_sft.jsonl"
    metrics = pipeline_runner.build(config).run()["stages"]["validate"]["metrics"]
    assert 0.0 <= metrics["tool_call_exact"] <= 1.0 and metrics["loss"] > 0

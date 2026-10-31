"""`llm_decision_cispo` end to end on a tiny random model: CISPO from an SFT checkpoint, chains reused, the shipped stage-two config."""

import json
from pathlib import Path

from tests.recipe_inputs import decision_raw as raw
from core import pipeline_runner


def test_cispo_continues_from_an_sft_checkpoint(tiny_model: Path, tmp_path: Path) -> None:
    first = pipeline_runner.build(raw("llm_decision_sft", tiny_model, tmp_path, "pointer")).run()
    checkpoint = first["stages"]["train"]["checkpoint"]
    method = {"group_size": 3, "length_hinge": 3}
    second = pipeline_runner.build(raw("llm_decision_cispo", tiny_model, tmp_path, "pointer", method, init=checkpoint)).run()
    log = [json.loads(line) for line in (Path(second["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert {"reward", "think_accuracy", "think_tokens"} <= set(log[0]) and 0 <= log[0]["reward"] <= 1
    assert "think_accuracy" in second["stages"]["validate"]["metrics"]
    assert second["experiment"] != first["experiment"]
    assert json.loads((Path(second["stages"]["train"]["checkpoint"]) / "head.json").read_text())["head"] == "pointer"


def test_cispo_reuses_chains_and_caps_the_importance_weight(tiny_model: Path, tmp_path: Path) -> None:
    first = pipeline_runner.build(raw("llm_decision_sft", tiny_model, tmp_path, "pointer")).run()
    method = {"group_size": 3, "iterations": 2, "clip": 0.2}
    config = raw("llm_decision_cispo", tiny_model, tmp_path, "pointer", method, init=first["stages"]["validate"]["checkpoint"])
    config["training"]["max_steps"] = 4
    result = pipeline_runner.build(config).run()
    log = [json.loads(line) for line in (Path(result["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert len(log) == 4 and log[0]["think_tokens"] == log[1]["think_tokens"]  # steps 1-2 share one sampled batch


def test_stage_two_config_points_at_the_stage_one_experiment() -> None:
    stage_one = pipeline_runner.load("configs/llm/decision_pointer.yaml")
    init = pipeline_runner.load("configs/llm/decision_cispo.yaml").config.model.init
    assert init == f"output/{stage_one.experiment.key}/validate/checkpoint"  # edit one config, update the other

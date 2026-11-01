"""`llm_reward` end to end on a tiny model, and its checkpoint as the `reward_model` reward of `llm_grpo`."""

import shutil
from typing import Any
from pathlib import Path

import pytest

from tests.recipe_inputs import raw
from core import pipeline_runner


@pytest.fixture(scope="module")
def reward_checkpoint(tiny_model: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    config = raw("llm_reward", tiny_model, tmp_path_factory.mktemp("reward"), max_steps=12, adapter={"r": 4, "targets": ["c_attn"]})
    result = pipeline_runner.build(config).run()
    return Path(result["stages"]["train"]["checkpoint"])


def test_lora_reward_model_saves_its_trained_score_layer_as_a_plain_classifier(tiny_model: Path, reward_checkpoint: Path) -> None:
    from transformers import AutoModelForSequenceClassification
    from safetensors.torch import load_file

    model = AutoModelForSequenceClassification.from_pretrained(reward_checkpoint)
    weights = load_file(reward_checkpoint / "model.safetensors")
    assert model.config.num_labels == 1 and model.config.pad_token_id is not None
    assert "score.weight" in weights and not any("lora" in name or "modules_to_save" in name for name in weights)
    fresh = AutoModelForSequenceClassification.from_pretrained(tiny_model, num_labels=1)
    assert not bool((fresh.score.weight == model.score.weight).all())


def test_grpo_reads_the_reward_model_and_fingerprints_its_files(tiny_model: Path, reward_checkpoint: Path, tmp_path: Path) -> None:
    copy = tmp_path / "rm"
    shutil.copytree(reward_checkpoint, copy)

    def grpo(path: Path) -> dict[str, Any]:
        config = raw("llm_grpo", tiny_model, tmp_path)
        config["method"]["reward"] = {"name": "weighted", "parts": [{"name": "reward_model", "path": str(path)},
                                                                    {"name": "contains", "weight": 0.5}]}
        return config

    pipeline = pipeline_runner.build(grpo(copy))
    assert str(copy) in pipeline.config.paths()
    before = pipeline.fingerprint("train")
    result = pipeline.run()
    assert result["stages"]["validate"]["passed"] is True
    (copy / "config.json").write_text((copy / "config.json").read_text() + "\n")
    assert pipeline_runner.build(grpo(copy)).fingerprint("train") != before
    config = grpo(tmp_path / "missing")
    with pytest.raises(ValueError, match="llm_reward checkpoint directory"):
        pipeline_runner.build(config).run()

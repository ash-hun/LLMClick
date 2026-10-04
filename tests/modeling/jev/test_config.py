from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from core.config.experiment import config_hash
from core import pipeline

CONFIGS = sorted(Path("configs/jev").glob("*.yaml"))


def changed(config: dict[str, Any], section: str, **keys: Any) -> dict[str, Any]:
    return {**config, section: {**config[section], **keys}}


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_configs_validate(path: Path) -> None:
    assert pipeline.load(path).kind == "jev"


def test_hash_is_stable_and_ignores_key_order(local_config: dict) -> None:
    a = pipeline.build(local_config).config
    b = pipeline.build(dict(reversed(list(local_config.items())))).config
    assert config_hash(a) == config_hash(b) and len(config_hash(a)) == 8


def test_recipe_is_required(local_config: dict) -> None:
    with pytest.raises(ValueError, match="pipeline.recipe"):
        pipeline.build(changed(local_config, "pipeline", recipe=None))


def test_unknown_registry_keys_are_rejected(local_config: dict) -> None:
    with pytest.raises(ValidationError, match="Unknown backbone"):
        pipeline.build(changed(local_config, "model", backbone="nope"))


def test_prompt_layout_requires_supporting_backbone(local_config: dict) -> None:
    with pytest.raises(ValidationError, match="prompt_layout"):
        pipeline.build(changed(local_config, "model", backbone="gemma4", prompt_layout="live-last"))


def test_revision_must_be_commit(local_config: dict) -> None:
    with pytest.raises(ValidationError, match="40-character"):
        pipeline.build(changed(local_config, "model", revision="main"))


def test_registries_are_listed() -> None:
    names = pipeline.RECIPES.get("jev").catalogue()
    assert {"local_jsonl", "huggingface", "jeff_extra", "jeff_probability"} <= set(names["builder"])
    assert {"example", "classification", "boolean"} <= set(names["converter"])
    assert {"qwen3_5", "gemma4", "modernbert"} <= set(names["backbone"])
    assert "openai_compatible" in names["teacher"]
    assert {"local_jsonl", "huggingface", "jeff_panel", "jeff_jevbench_hard"} <= set(names["benchmark"])

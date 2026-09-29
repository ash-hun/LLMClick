from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config.experiment import config_hash, load_config
from core.config.schema import PipelineConfig

CONFIGS = sorted(Path("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_shipped_configs_validate(path: Path) -> None:
    config = load_config(path)
    assert config.name


def test_hash_is_stable_and_ignores_key_order(local_config: dict) -> None:
    pipeline = local_config["pipeline"]
    a = PipelineConfig(**{**pipeline, **{k: v for k, v in local_config.items() if k != "pipeline"}})
    reordered = dict(reversed(list(local_config.items())))
    b = PipelineConfig(**{**pipeline, **{k: v for k, v in reordered.items() if k != "pipeline"}})
    assert config_hash(a) == config_hash(b) and len(config_hash(a)) == 8


def test_unknown_registry_keys_are_rejected(local_config: dict) -> None:
    bad = {**local_config, "model": {**local_config["model"], "backbone": "nope"}}
    with pytest.raises(ValidationError, match="Unknown backbone"):
        PipelineConfig(**bad["pipeline"], **{k: v for k, v in bad.items() if k != "pipeline"})


def test_prompt_layout_requires_supporting_backbone(local_config: dict) -> None:
    bad = {**local_config, "model": {**local_config["model"], "backbone": "gemma4", "prompt_layout": "live-last"}}
    with pytest.raises(ValidationError, match="prompt_layout"):
        PipelineConfig(**bad["pipeline"], **{k: v for k, v in bad.items() if k != "pipeline"})


def test_revision_must_be_commit(local_config: dict) -> None:
    bad = {**local_config, "model": {**local_config["model"], "revision": "main"}}
    with pytest.raises(ValidationError, match="40-character"):
        PipelineConfig(**bad["pipeline"], **{k: v for k, v in bad.items() if k != "pipeline"})

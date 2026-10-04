"""Every tuning recipe end to end on a tiny random model: data -> train -> validate, then a rerun that builds nothing."""

import json
from pathlib import Path
from typing import Any

import pytest

from core.progress import StateProgress
from core import pipeline

RECIPES = {  # recipe -> (sample file, method section, metric the validate stage must report)
    "llm_sft": ("llm_sft", {}, "loss"),
    "llm_instruction": ("llm_instruction", {"system": "Add."}, "perplexity"),
    "llm_dpo": ("llm_dpo", {"beta": 0.1}, "accuracy"),
    "llm_grpo": ("llm_grpo", {"group_size": 3, "max_new_tokens": 3, "reward": {"name": "contains"}}, "reward"),
    "embedding_contrastive": ("embedding_contrastive", {"temperature": 0.1}, "mrr"),
}


def raw(recipe: str, model: Path, out: Path, **training: Any) -> dict[str, Any]:
    sample, method, _ = RECIPES[recipe]
    return {
        "pipeline": {"recipe": recipe, "name": "tiny", "seed": 3, "output_dir": str(out), "device": "cpu"},
        "model": {"architecture": "bi_encoder" if recipe.startswith("embedding") else "transformer", "name": str(model)},
        "data": {"sources": [{"name": "local_jsonl", "path": f"samples/{sample}.jsonl"}], "validation": 0.2},
        "method": method,
        "training": {"lr": 1e-3, "batch_size": 4, "max_length": 64, "max_steps": 3, **training},
    }


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_recipe_trains_validates_and_reruns_from_cache(recipe: str, tiny_model: Path, tmp_path: Path) -> None:
    first = pipeline.build(raw(recipe, tiny_model, tmp_path)).run()
    stages = first["stages"]
    assert stages["data"]["rows"] == {"train": 64, "validation": 16}
    assert stages["train"]["steps"] == 3 and (Path(stages["train"]["checkpoint"]) / "config.json").exists()
    assert stages["validate"]["passed"] is True and RECIPES[recipe][2] in stages["validate"]["metrics"]
    progress = StateProgress()
    second = pipeline.build(raw(recipe, tiny_model, tmp_path), progress).run()
    assert json.dumps(second) == json.dumps(first)
    assert progress.snapshot()["stages"] == {"data": "cached", "train": "cached", "validate": "cached"}


def test_sft_learns_the_sample_rows(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_sft", tiny_model, tmp_path, max_steps=None, epochs=3, lr=3e-3)
    result = pipeline.build(config).run()
    log = [json.loads(line) for line in (Path(result["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert [event["step"] for event in log] == list(range(1, 49)) and log[-1]["loss"] < log[0]["loss"] / 2


def test_training_change_reuses_the_data_stage(tiny_model: Path, tmp_path: Path) -> None:
    a = pipeline.build(raw("llm_sft", tiny_model, tmp_path))
    b = pipeline.build(raw("llm_sft", tiny_model, tmp_path, lr=5e-4))
    assert a.fingerprint("data") == b.fingerprint("data") and a.fingerprint("train") != b.fingerprint("train")


def test_row_that_does_not_fit_the_recipe_is_named(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_dpo", tiny_model, tmp_path)
    config["data"]["sources"][0]["path"] = "samples/llm_sft.jsonl"
    with pytest.raises(ValueError, match="Row 0 does not fit recipe 'llm_dpo'"):
        pipeline.build(config).run()


def test_failed_bound_stops_after_training(tiny_model: Path, tmp_path: Path) -> None:
    config = {**raw("llm_sft", tiny_model, tmp_path), "validation": {"max": {"loss": 0.0001}}}
    with pytest.raises(RuntimeError, match="above the maximum"):
        pipeline.build(config).run()


def test_config_rules(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_sft", tiny_model, tmp_path)
    with pytest.raises(ValueError, match="40-character commit"):
        pipeline.build({**config, "model": {"architecture": "transformer", "name": "Qwen/Qwen3-0.6B"}})
    with pytest.raises(ValueError, match="Unknown architecture 'bi_encoder'"):
        pipeline.build({**config, "model": {**config["model"], "architecture": "bi_encoder"}})
    with pytest.raises(ValueError, match="Unknown reward"):
        pipeline.build({**raw("llm_grpo", tiny_model, tmp_path), "method": {"reward": {"name": "nope"}}})


SHIPPED = sorted([*Path("configs/llm").glob("*.yaml"), *Path("configs/embedding").glob("*.yaml")])


@pytest.mark.parametrize("path", SHIPPED, ids=[p.name for p in SHIPPED])
def test_shipped_configs_validate(path: Path) -> None:
    assert pipeline.load(path).plan() == ["data", "train", "validate"]

"""`llm_merge` end to end on tiny fine-tunes: exact soups, TIES and DARE, the rules on what may be merged."""

import json
import shutil
from typing import Any
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file

from tests.recipe_inputs import raw
from core import pipeline_runner


@pytest.fixture(scope="module")
def finetunes(tiny_model: Path, tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    out = tmp_path_factory.mktemp("finetunes")
    paths = []
    for seed in (3, 4):
        config = raw("llm_sft", tiny_model, out, max_steps=6)
        config["pipeline"]["seed"] = seed
        paths.append(Path(pipeline_runner.build(config).run()["stages"]["train"]["checkpoint"]))
    return paths


def merge_config(tiny_model: Path, out: Path, rule: str, models: list[Any], **keys: Any) -> dict[str, Any]:
    return {
        "pipeline": {"recipe": "llm_merge", "name": "tiny", "seed": 3, "output_dir": str(out), "device": "cpu"},
        "model": {"architecture": "transformer", "name": str(tiny_model)},
        "data": {"sources": [{"name": "local_jsonl", "path": "samples/llm_sft.jsonl"}], "validation": 0.2},
        "training": {"max_length": 64},
        "merge": {"rule": rule, "models": [m if isinstance(m, dict) else {"path": str(m)} for m in models], **keys},
    }


def weights(result: dict[str, Any]) -> dict[str, Any]:
    return load_file(Path(result["stages"]["train"]["checkpoint"]) / "model.safetensors")


def test_a_soup_of_one_model_with_itself_is_that_model(tiny_model: Path, finetunes: list[Path], tmp_path: Path) -> None:
    first = pipeline_runner.build(merge_config(tiny_model, tmp_path, "linear", [finetunes[0], finetunes[0]])).run()
    merged, source = weights(first), load_file(finetunes[0] / "model.safetensors")
    assert merged.keys() == source.keys() and all(bool((merged[k] == source[k]).all()) for k in source)
    checkpoint = Path(first["stages"]["train"]["checkpoint"])
    assert (checkpoint / "config.json").is_file() and (checkpoint / "tokenizer.json").is_file()
    metrics = first["stages"]["validate"]["metrics"]
    assert set(metrics) >= {"loss", "base_loss", "loss_delta"} and metrics["loss_delta"] < 0
    second = pipeline_runner.build(merge_config(tiny_model, tmp_path, "linear", [finetunes[0], finetunes[0]])).run()
    assert json.dumps(second) == json.dumps(first)


@pytest.mark.parametrize("rule", ["ties", "dare_ties", "task_arithmetic", "slerp"])
def test_rules_merge_two_fine_tunes_into_a_third_model(rule: str, tiny_model: Path, finetunes: list[Path], tmp_path: Path) -> None:
    result = pipeline_runner.build(merge_config(tiny_model, tmp_path, rule, finetunes, density=0.5)).run()
    merged = weights(result)
    sources = [load_file(path / "model.safetensors") for path in finetunes]
    name = "transformer.h.0.attn.c_attn.weight"
    assert all(not bool((merged[name] == source[name]).all()) for source in sources)
    assert result["stages"]["validate"]["passed"] is True


def test_dare_drops_by_the_seed(tiny_model: Path, finetunes: list[Path], tmp_path: Path) -> None:
    def run(out: str, seed: int) -> dict[str, Any]:
        config = merge_config(tiny_model, tmp_path / out, "dare_linear", finetunes, density=0.3)
        config["pipeline"]["seed"] = seed
        return weights(pipeline_runner.build(config).run())

    first, again, other = run("a", 3), run("b", 3), run("c", 4)
    name = "transformer.h.0.attn.c_attn.weight"
    assert bool((first[name] == again[name]).all()) and not bool((first[name] == other[name]).all())


def test_merge_rules(tiny_model: Path, finetunes: list[Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        pipeline_runner.build(merge_config(tiny_model, tmp_path, "slerp", [*finetunes, finetunes[0]]))
    with pytest.raises(ValueError, match="at least two"):
        pipeline_runner.build(merge_config(tiny_model, tmp_path, "linear", finetunes[:1]))
    with pytest.raises(ValueError, match="Unknown"):
        pipeline_runner.build(merge_config(tiny_model, tmp_path, "average", finetunes))
    with pytest.raises(ValueError, match="40-character"):
        pipeline_runner.build(merge_config(tiny_model, tmp_path, "linear", [*finetunes, {"name": "Qwen/Qwen3-0.6B"}]))
    with pytest.raises(ValueError, match="not both"):
        pipeline_runner.build(merge_config(tiny_model, tmp_path, "linear", [*finetunes, {"path": "x", "name": "y", "revision": "a" * 40}]))
    with pytest.raises(FileNotFoundError, match="run the experiment"):
        pipeline_runner.build(merge_config(tiny_model, tmp_path, "linear", [finetunes[0], tmp_path / "missing"])).run()


def test_models_of_another_shape_or_quantized_are_refused(tiny_model: Path, finetunes: list[Path], tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    shutil.copytree(finetunes[1], extra)
    tensors = load_file(extra / "model.safetensors")
    save_file({**tensors, "added.weight": tensors["transformer.wte.weight"][:2].clone()}, str(extra / "model.safetensors"))
    with pytest.raises(ValueError, match="other tensors"):
        pipeline_runner.build(merge_config(tiny_model, tmp_path / "a", "linear", [finetunes[0], extra])).run()
    quantized = tmp_path / "quantized"
    shutil.copytree(finetunes[1], quantized)
    config = json.loads((quantized / "config.json").read_text())
    (quantized / "config.json").write_text(json.dumps({**config, "quantization_config": {"quant_method": "awq"}}))
    with pytest.raises(ValueError, match="quantized weights"):
        pipeline_runner.build(merge_config(tiny_model, tmp_path / "b", "linear", [finetunes[0], quantized])).run()


def test_a_merged_checkpoint_counts_by_content(tiny_model: Path, finetunes: list[Path], tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(finetunes[1], copy)

    def fingerprint() -> str:
        return pipeline_runner.build(merge_config(tiny_model, tmp_path, "linear", [finetunes[0], copy])).fingerprint("train")

    before = fingerprint()
    (copy / "config.json").write_text((copy / "config.json").read_text() + "\n")
    assert fingerprint() != before

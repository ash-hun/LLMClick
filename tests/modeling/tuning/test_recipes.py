"""Every tuning recipe end to end on a tiny random model: data -> train -> validate, then a rerun that builds nothing."""

import json
from pathlib import Path
from typing import Any

import pytest

from core.progress import Progress, StateProgress
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
        "method": dict(method),  # a copy: tests change it, and RECIPES is shared
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


@pytest.mark.parametrize("recipe", ["llm_sft", "llm_dpo", "embedding_contrastive"])
def test_lora_trains_only_the_adapter_and_saves_a_merged_checkpoint(recipe: str, tiny_model: Path, tmp_path: Path) -> None:
    from safetensors.torch import load_file
    adapter = {"name": "lora", "r": 4, "alpha": 8, "targets": ["c_attn"]}  # GPT-2 names its attention projection c_attn
    result = pipeline.build(raw(recipe, tiny_model, tmp_path, adapter=adapter)).run()
    assert result["stages"]["validate"]["passed"] is True
    base = load_file(tiny_model / "model.safetensors")
    tuned = load_file(Path(result["stages"]["train"]["checkpoint"]) / "model.safetensors")

    def key(name: str) -> str:  # the base file is a causal LM; the embedding checkpoint is the bare transformer
        return name if name in tuned else name.removeprefix("transformer.")

    assert not any("lora" in name for name in tuned)  # merged: loads without peft
    changed = {name for name in base if key(name) in tuned and not bool((base[name] == tuned[key(name)]).all())}
    assert changed and all("c_attn.weight" in name for name in changed)


def test_adapter_config_rules(tiny_model: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        pipeline.build(raw("llm_sft", tiny_model, tmp_path, adapter={"rank": 4}))
    full = pipeline.build(raw("llm_sft", tiny_model, tmp_path))
    lora = pipeline.build(raw("llm_sft", tiny_model, tmp_path, adapter={"targets": ["c_attn"]}))
    assert full.fingerprint("train") != lora.fingerprint("train") and full.fingerprint("data") == lora.fingerprint("data")
    with pytest.raises(ValueError, match="Target modules"):
        pipeline.build(raw("llm_sft", tiny_model, tmp_path, adapter={})).run()  # GPT-2 has no q_proj


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


@pytest.mark.parametrize("section, key", [("training", "learning_rate"), ("method", "bta"), ("data", "validaton"),
                                          ("validation", "mn"), ("model", "revison"), ("tracker", "enable")])
def test_misspelt_keys_inside_sections_are_rejected(section: str, key: str, tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_dpo", tiny_model, tmp_path)
    config[section] = {**config.get(section, {}), key: 1}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        pipeline.build(config)


def test_changing_what_the_row_check_reads_checks_the_rows_again(tiny_model: Path, tmp_path: Path) -> None:
    exact = pipeline.build(raw("llm_grpo", tiny_model, tmp_path))
    config = raw("llm_grpo", tiny_model, tmp_path)
    config["method"]["reward"] = {"name": "last_number"}
    assert exact.fingerprint("data") != pipeline.build(config).fingerprint("data")
    config = raw("llm_grpo", tiny_model, tmp_path)
    config["method"]["group_size"] = 5  # not read by the row check: the data stage is shared
    assert exact.fingerprint("data") == pipeline.build(config).fingerprint("data")


def test_initial_checkpoint_counts_by_content_not_by_path(tiny_model: Path, tmp_path: Path) -> None:
    first, second, link = tmp_path / "first", tmp_path / "second", tmp_path / "current"
    for directory, text in ((first, "a"), (second, "b")):
        directory.mkdir()
        (directory / "config.json").write_text(text)
    link.symlink_to(first)

    def fingerprint() -> str:
        config = raw("llm_sft", tiny_model, tmp_path)
        config["model"]["init"] = str(link)
        return pipeline.build(config).fingerprint("train")

    before = fingerprint()
    assert fingerprint() == before
    link.unlink()
    link.symlink_to(second)  # the same path now leads to another build, as after rebuilding an earlier experiment
    repointed = fingerprint()
    (second / "config.json").write_text("c")
    assert len({before, repointed, fingerprint()}) == 3


SHIPPED = sorted([*Path("configs/llm").glob("*.yaml"), *Path("configs/embedding").glob("*.yaml")])


@pytest.mark.parametrize("path", SHIPPED, ids=[p.name for p in SHIPPED])
def test_shipped_configs_validate(path: Path) -> None:
    assert pipeline.load(path).plan() == ["data", "train", "validate"]


class Notes(Progress):
    """Collects every phase a stage reports."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def update(self, done: int, total: int | None = None, note: str = "") -> None:
        if note and note not in self.seen:
            self.seen.append(note)


def with_long_rows(tmp_path: Path, count: int) -> str:
    rows = [json.loads(line) for line in Path("samples/llm_sft.jsonl").read_text().splitlines()]
    for row in rows[:count]:
        row["messages"][0]["content"] = " ".join(["What is 1 plus 2 ?"] * 12)  # about 70 tokens
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def test_rows_over_max_length_stop_the_run_before_training(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_sft", tiny_model, tmp_path, max_length=40)
    config["data"]["sources"][0]["path"] = with_long_rows(tmp_path, 5)
    notes = Notes()
    with pytest.raises(ValueError, match="rows need more than training.max_length=40"):
        pipeline.build(config, notes).run()
    assert not list(tmp_path.glob("_stages/train-*/training.jsonl"))  # no step was taken
    assert "loading model" in notes.seen and "sampling" not in " ".join(notes.seen)


def test_overflow_skip_leaves_long_rows_out_of_training_and_validation(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_sft", tiny_model, tmp_path, max_length=40, overflow="skip")
    config["data"]["sources"][0]["path"] = with_long_rows(tmp_path, 5)
    result = pipeline.build(config).run()
    data, train, validate = (result["stages"][name] for name in ("data", "train", "validate"))
    assert data["rows"] == {"train": 64, "validation": 16} and train["rows"] < 64 and validate["passed"] is True
    held_out = [json.loads(line) for line in Path(data["validation"]).read_text().splitlines()]
    long_held_out = sum(len(row["messages"][0]["content"]) > 100 for row in held_out)
    assert train["rows"] == 64 - (5 - long_held_out)
    every = raw("llm_sft", tiny_model, tmp_path, max_length=8, overflow="skip")
    with pytest.raises(ValueError, match="rows need more than training.max_length=8"):
        pipeline.build(every).run()  # skipping everything is still an error


def test_every_long_phase_reports_progress(tiny_model: Path, tmp_path: Path) -> None:
    notes = Notes()
    pipeline.build(raw("llm_dpo", tiny_model, tmp_path), notes).run()
    for phase in ("loading model", "checking validation lengths", "checking training lengths", "reference margins",
                  "saving checkpoint", "loading checkpoint"):
        assert phase in notes.seen, (phase, notes.seen)

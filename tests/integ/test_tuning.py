"""The tuning family (`llm_sft`, `llm_instruction`, `llm_dpo`, `llm_grpo`, `embedding_contrastive`) end to end: what every recipe shares."""

import json
from typing import Any
from pathlib import Path

import pytest

from core.progress_reporter import Progress, StateProgress
from tests.recipe_inputs import RECIPES, raw
from tests.recipe_fakes import FakeWandb
from core import pipeline_runner

SHIPPED = sorted([*Path("configs/llm").glob("*.yaml"), *Path("configs/embedding").glob("*.yaml")])


@pytest.mark.parametrize("recipe", sorted(RECIPES))
def test_recipe_trains_validates_and_reruns_from_cache(recipe: str, tiny_model: Path, tmp_path: Path) -> None:
    first = pipeline_runner.build(raw(recipe, tiny_model, tmp_path)).run()
    stages = first["stages"]
    assert stages["data"]["rows"] == {"train": 64, "validation": 16}
    assert stages["train"]["steps"] == 3 and (Path(stages["train"]["checkpoint"]) / "config.json").exists()
    assert stages["validate"]["passed"] is True and RECIPES[recipe][2] in stages["validate"]["metrics"]
    progress = StateProgress()
    second = pipeline_runner.build(raw(recipe, tiny_model, tmp_path), progress).run()
    assert json.dumps(second) == json.dumps(first)
    assert progress.snapshot()["stages"] == {"data": "cached", "train": "cached", "validate": "cached"}


@pytest.mark.parametrize("recipe", ["llm_sft", "llm_dpo", "embedding_contrastive"])
def test_lora_trains_only_the_adapter_and_saves_a_merged_checkpoint(recipe: str, tiny_model: Path, tmp_path: Path) -> None:
    from safetensors.torch import load_file
    adapter = {"name": "lora", "r": 4, "alpha": 8, "targets": ["c_attn"]}  # GPT-2 names its attention projection c_attn
    result = pipeline_runner.build(raw(recipe, tiny_model, tmp_path, adapter=adapter)).run()
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
        pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path, adapter={"rank": 4}))
    full = pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path))
    lora = pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path, adapter={"targets": ["c_attn"]}))
    assert full.fingerprint("train") != lora.fingerprint("train") and full.fingerprint("data") == lora.fingerprint("data")
    with pytest.raises(ValueError, match="Target modules"):
        pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path, adapter={})).run()  # GPT-2 has no q_proj


def test_training_change_reuses_the_data_stage(tiny_model: Path, tmp_path: Path) -> None:
    a = pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path))
    b = pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path, lr=5e-4))
    assert a.fingerprint("data") == b.fingerprint("data") and a.fingerprint("train") != b.fingerprint("train")


def test_row_that_does_not_fit_the_recipe_is_named(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_dpo", tiny_model, tmp_path)
    config["data"]["sources"][0]["path"] = "samples/llm_sft.jsonl"
    with pytest.raises(ValueError, match="Row 0 does not fit recipe 'llm_dpo'"):
        pipeline_runner.build(config).run()


def test_failed_bound_stops_after_training(tiny_model: Path, tmp_path: Path) -> None:
    config = {**raw("llm_sft", tiny_model, tmp_path), "validation": {"max": {"loss": 0.0001}}}
    with pytest.raises(RuntimeError, match="above the maximum"):
        pipeline_runner.build(config).run()


def test_config_rules(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_sft", tiny_model, tmp_path)
    with pytest.raises(ValueError, match="40-character commit"):
        pipeline_runner.build({**config, "model": {"architecture": "transformer", "name": "Qwen/Qwen3-0.6B"}})
    with pytest.raises(ValueError, match="Unknown architecture 'bi_encoder'"):
        pipeline_runner.build({**config, "model": {**config["model"], "architecture": "bi_encoder"}})
    with pytest.raises(ValueError, match="Unknown reward"):
        pipeline_runner.build({**raw("llm_grpo", tiny_model, tmp_path), "method": {"reward": {"name": "nope"}}})


@pytest.mark.parametrize("section, key", [("training", "learning_rate"), ("method", "bta"), ("data", "validaton"),
                                          ("validation", "mn"), ("model", "revison"), ("tracker", "enable")])
def test_misspelt_keys_inside_sections_are_rejected(section: str, key: str, tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_dpo", tiny_model, tmp_path)
    config[section] = {**config.get(section, {}), key: 1}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        pipeline_runner.build(config)


def test_initial_checkpoint_counts_by_content_not_by_path(tiny_model: Path, tmp_path: Path) -> None:
    first, second, link = tmp_path / "first", tmp_path / "second", tmp_path / "current"
    for directory, text in ((first, "a"), (second, "b")):
        directory.mkdir()
        (directory / "config.json").write_text(text)
    link.symlink_to(first)

    def fingerprint() -> str:
        config = raw("llm_sft", tiny_model, tmp_path)
        config["model"]["init"] = str(link)
        return pipeline_runner.build(config).fingerprint("train")

    before = fingerprint()
    assert fingerprint() == before
    link.unlink()
    link.symlink_to(second)  # the same path now leads to another build, as after rebuilding an earlier experiment
    repointed = fingerprint()
    (second / "config.json").write_text("c")
    assert len({before, repointed, fingerprint()}) == 3


@pytest.mark.parametrize("path", SHIPPED, ids=[p.name for p in SHIPPED])
def test_shipped_configs_validate(path: Path) -> None:
    assert pipeline_runner.load(path).plan() == ["data", "train", "validate"]


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
        pipeline_runner.build(config, notes).run()
    assert not list(tmp_path.glob("_stages/train-*/training.jsonl"))  # no step was taken
    assert "loading model" in notes.seen and "sampling" not in " ".join(notes.seen)


def test_overflow_skip_leaves_long_rows_out_of_training_and_validation(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_sft", tiny_model, tmp_path, max_length=40, overflow="skip")
    config["data"]["sources"][0]["path"] = with_long_rows(tmp_path, 5)
    result = pipeline_runner.build(config).run()
    data, train, validate = (result["stages"][name] for name in ("data", "train", "validate"))
    assert data["rows"] == {"train": 64, "validation": 16} and train["rows"] < 64 and validate["passed"] is True
    held_out = [json.loads(line) for line in Path(data["validation"]).read_text().splitlines()]
    long_held_out = sum(len(row["messages"][0]["content"]) > 100 for row in held_out)
    assert train["rows"] == 64 - (5 - long_held_out)
    every = raw("llm_sft", tiny_model, tmp_path, max_length=8, overflow="skip")
    with pytest.raises(ValueError, match="rows need more than training.max_length=8"):
        pipeline_runner.build(every).run()  # skipping everything is still an error


def test_every_long_phase_reports_progress(tiny_model: Path, tmp_path: Path) -> None:
    notes = Notes()
    pipeline_runner.build(raw("llm_dpo", tiny_model, tmp_path), notes).run()
    for phase in ("loading model", "checking validation lengths", "checking training lengths", "reference margins",
                  "saving checkpoint", "loading checkpoint"):
        assert phase in notes.seen, (phase, notes.seen)


def test_shared_model_keys_stay_out_of_other_families(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("embedding_contrastive", tiny_model, tmp_path)
    for key, value in (("template", {"enable_thinking": False}), ("head", "pointer")):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            pipeline_runner.build({**config, "model": {**config["model"], key: value}})
    llm = raw("llm_sft", tiny_model, tmp_path)
    assert pipeline_runner.build({**llm, "model": {**llm["model"], "template": {"enable_thinking": False}}})


def test_bumping_one_parts_version_rebuilds_only_its_recipes(tiny_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from modeling.llm.methods.dpo_method import DPO

    def fingerprints() -> dict[str, str]:
        return {name: pipeline_runner.build(raw(name, tiny_model, tmp_path)).fingerprint("train") for name in ("llm_dpo", "llm_sft")}

    before = fingerprints()
    monkeypatch.setattr(DPO, "version", DPO.version + 1)
    after = fingerprints()
    assert after["llm_dpo"] != before["llm_dpo"] and after["llm_sft"] == before["llm_sft"]


def test_tracker_is_synced_once_even_when_switched_on_after_training(tiny_model: Path, tmp_path: Path, wandb: FakeWandb) -> None:
    plain = pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path)).run()
    assert wandb.runs == []  # tracker off
    tracked = {**raw("llm_sft", tiny_model, tmp_path), "tracker": {"enabled": True, "project": "p"}}
    again = pipeline_runner.build(tracked).run()  # same experiment, training comes from cache: the history is replayed
    assert again["experiment"] == plain["experiment"] and len(wandb.runs) == 1 and wandb.steps == [1, 2, 3]
    assert wandb.runs[0]["id"] == Path(plain["stages"]["train"]["run"]).name
    pipeline_runner.build(tracked).run()
    assert len(wandb.runs) == 1  # already sent to this project
    pipeline_runner.build({**tracked, "tracker": {"enabled": True, "project": "other"}}).run()
    assert [run["project"] for run in wandb.runs] == ["p", "other"]


def test_tracker_logs_each_step_while_training_runs(tiny_model: Path, tmp_path: Path, wandb: FakeWandb,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from modeling.llm.methods.sft_method import SFT

    seen_during_training: list[int] = []
    original = SFT.loss

    def watching(self: Any, backbone: Any, rows: Any) -> Any:
        seen_during_training.append(len(wandb.steps))  # steps already sent when the next one starts
        return original(self, backbone, rows)

    monkeypatch.setattr(SFT, "loss", watching)
    tracked = {**raw("llm_sft", tiny_model, tmp_path), "tracker": {"enabled": True, "project": "p"}}
    result = pipeline_runner.build(tracked).run()
    assert seen_during_training == [0, 1, 2] and wandb.steps == [1, 2, 3] and wandb.finished == 1
    marker = json.loads((Path(result["stages"]["train"]["run"]) / "tracker.json").read_text())
    assert marker == {"synced": ["/p"]}
    pipeline_runner.build(tracked).run()
    assert len(wandb.runs) == 1 and wandb.steps == [1, 2, 3]  # nothing is replayed on top of the live run


def test_interrupted_tracked_training_is_not_marked_as_sent(tiny_model: Path, tmp_path: Path, wandb: FakeWandb,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from modeling.llm.methods.sft_method import SFT

    original, calls = SFT.loss, {"n": 0}

    def failing(self: Any, backbone: Any, rows: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return original(self, backbone, rows)

    monkeypatch.setattr(SFT, "loss", failing)
    tracked = {**raw("llm_sft", tiny_model, tmp_path), "tracker": {"enabled": True, "project": "p"}}
    with pytest.raises(KeyboardInterrupt):
        pipeline_runner.build(tracked).run()
    assert wandb.finished == 1 and not list(tmp_path.glob("_stages/train-*/tracker.json"))  # closed, not marked
    pipeline_runner.build(tracked).run()  # the rerun opens the same wandb run again and completes it
    assert len(wandb.runs) == 2 and wandb.runs[0]["id"] == wandb.runs[1]["id"] and wandb.runs[1]["resume"] == "allow"
    assert list(tmp_path.glob("_stages/train-*/tracker.json"))


def test_eval_every_records_held_out_metrics_in_the_history(tiny_model: Path, tmp_path: Path) -> None:
    result = pipeline_runner.build(raw("llm_dpo", tiny_model, tmp_path, eval_every=3, eval_rows=8)).run()
    log = [json.loads(line) for line in (Path(result["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert [sorted(k for k in event if k.startswith("eval/")) for event in log] == [[], [], ["eval/accuracy", "eval/margin"]]
    stage = pipeline_runner.build(raw("llm_dpo", tiny_model, tmp_path, eval_every=3, eval_rows=8)).stages["train"]
    assert stage.history(result["stages"]["train"])[-1][1].keys() >= {"train/loss", "train/accuracy", "eval/accuracy"}

"""`llm_decision_sft` end to end on a tiny random model: both heads trained, calibrated and reloaded, reasoning sampled once."""

import json
from pathlib import Path

import pytest

from tests.recipe_inputs import DECISION_ROWS as ROWS, decision_raw as raw
from modeling.llm.models.transformer import TransformerBackbone
from modeling.llm.models.config import LLMBackboneConfig
from core import pipeline


@pytest.mark.parametrize("head", ["pointer", "readout"])
def test_decision_sft_trains_calibrates_and_reloads_its_head(head: str, tiny_model: Path, tmp_path: Path) -> None:
    result = pipeline.build(raw("llm_decision_sft", tiny_model, tmp_path, head)).run()
    train, validate = result["stages"]["train"], result["stages"]["validate"]
    checkpoint = Path(train["checkpoint"])
    saved = json.loads((checkpoint / "head.json").read_text())
    assert saved["head"] == head and saved["temperature"] == train["temperature"] > 0
    assert (checkpoint / "head.safetensors").exists() and train["calibration_rows"] > 0
    assert {"accuracy", "nll", "ece", "questions"} <= set(validate["metrics"])
    assert validate["metrics"]["temperature"] == train["temperature"]  # validate read the head back from the checkpoint
    with pytest.raises(ValueError, match="trained with head"):
        other = "readout" if head == "pointer" else "pointer"
        backbone_config = LLMBackboneConfig(architecture="transformer", name=str(tiny_model), head=other)
        TransformerBackbone(backbone_config).load("cpu", checkpoint)


def test_reasoning_chains_are_sampled_once_and_reused(tiny_model: Path, tmp_path: Path) -> None:
    result = pipeline.build(raw("llm_decision_sft", tiny_model, tmp_path, "pointer", {"think_fraction": 0.5})).run()
    chains = json.loads((Path(result["stages"]["train"]["run"]) / "reasoning_chains.json").read_text())
    questions = sum(len(row["questions"]) for row in ROWS)
    assert 0.2 * questions < len(chains) < 0.6 * questions and all(len(chain) <= 6 for chain in chains.values())


def test_decision_phases_report_progress_and_count_reasoning_in_the_length(tiny_model: Path, tmp_path: Path) -> None:
    from core.progress import Progress

    class Notes(Progress):
        def __init__(self) -> None:
            self.seen: set[str] = set()

        def update(self, done: int, total: int | None = None, note: str = "") -> None:
            self.seen.add(note)

    notes = Notes()
    pipeline.build(raw("llm_decision_sft", tiny_model, tmp_path, "pointer", {"think_fraction": 0.5}), notes).run()
    assert {"sampling reasoning chains", "fitting temperature", "checking training lengths"} <= notes.seen
    tight = raw("llm_decision_sft", tiny_model, tmp_path, "pointer", {"think_fraction": 0.5, "max_think": 200})
    with pytest.raises(ValueError, match="need more than training.max_length=250"):
        pipeline.build(tight).run()  # the prompt fits, prompt plus 200 reasoning tokens does not


def test_decision_config_rules(tiny_model: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs one of"):
        pipeline.build(raw("llm_decision_sft", tiny_model, tmp_path, "nope"))
    plain = raw("llm_decision_sft", tiny_model, tmp_path, "pointer")
    with pytest.raises(ValueError, match="has no decision head"):
        pipeline.build({**plain, "pipeline": {**plain["pipeline"], "recipe": "llm_dpo"}, "method": {}})
    pending = pipeline.build(raw("llm_decision_cispo", tiny_model, tmp_path, "pointer", init=str(tmp_path / "missing")))
    with pytest.raises(FileNotFoundError, match="run the experiment that writes it"):
        pending.run()  # the config is valid before stage 1 has run; training is what needs the checkpoint
    a = pipeline.build(raw("llm_decision_sft", tiny_model, tmp_path, "pointer"))
    b = pipeline.build(raw("llm_decision_sft", tiny_model, tmp_path, "readout"))
    assert a.fingerprint("data") == b.fingerprint("data") and a.fingerprint("train") != b.fingerprint("train")


def test_batch_size_counts_questions_in_training(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_decision_sft", tiny_model, tmp_path, "pointer")
    config["training"].update(batch_size=5, max_steps=None)
    result = pipeline.build(config).run()
    train = result["stages"]["train"]
    records = result["stages"]["data"]["rows"]["train"] - int(train["calibration_rows"])
    assert train["rows"] == records * 3 and train["steps"] == -(-train["rows"] // 5)  # 3 questions per record

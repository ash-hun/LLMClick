"""The pieces under the recipes: target masks, the batch schedule, resume, DPO's starting point, pooling, rewards."""

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from modeling.embedding.models.bi_encoder import BiEncoder
from modeling.llm.methods.grpo import contains, exact_match, last_number_match
from modeling.llm.models.transformer import TransformerBackbone
from modeling.tuning.config import AdapterConfig, BackboneConfig, TrainingConfig
from modeling.llm.models.config import LLMBackboneConfig
from modeling.tuning.loop import fit, learning_rate, schedule, tracked
from modeling.llm.methods.dpo import DPO
from modeling.llm.methods.sft import SFT
from core.progress import Progress

ROWS = [json.loads(line) for line in Path("samples/llm_sft.jsonl").read_text().splitlines()][:12]


@pytest.fixture
def backbone(tiny_model: Path) -> TransformerBackbone:
    loaded = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    loaded.load("cpu")
    return loaded


def test_targets_are_exactly_the_assistant_turns(backbone: TransformerBackbone) -> None:
    messages = [{"role": "user", "content": "What is 2 plus 3 ?"}, {"role": "assistant", "content": "5"},
                {"role": "user", "content": "What is 4 plus 4 ?"}, {"role": "assistant", "content": "8"}]
    method = SFT(SFT.Config(), TrainingConfig())
    ids, targets = method.encode(backbone, messages)
    assert backbone.tokenizer.decode([i for i, flag in zip(ids, targets, strict=True) if flag]) == "5 <|im_end|> 8 <|im_end|>"
    ids, targets = method.encode(backbone, messages, last_only=True)
    assert backbone.tokenizer.decode([i for i, flag in zip(ids, targets, strict=True) if flag]) == "8 <|im_end|>"


def test_schedule_is_deterministic_and_covers_every_row() -> None:
    training = TrainingConfig(epochs=2, batch_size=4, accumulation=2)
    steps = schedule(10, training, seed=1)
    assert steps == schedule(10, training, seed=1) and len(steps) == 3  # 6 batches, 2 per step
    assert sorted(i for step in steps for batch in step for i in batch) == sorted(list(range(10)) * 2)
    assert len(schedule(10, TrainingConfig(batch_size=4, max_steps=1), seed=1)) == 1
    twice = schedule(10, TrainingConfig(batch_size=4), seed=1, repeats=2)
    assert len(twice) == 6 and twice[0] == twice[1] and twice[2] == twice[3] and twice[0] != twice[2]


def test_learning_rate_warms_up_then_decays() -> None:
    training = TrainingConfig(lr=1.0, warmup_ratio=0.2)
    rates = [learning_rate(step, 10, training) for step in range(10)]
    assert rates[:2] == [0.5, 1.0] and rates[2] == 1.0 and rates[-1] == pytest.approx(1 / 8) and rates[2:] == sorted(rates[2:], reverse=True)


@pytest.mark.parametrize("adapter", [None, AdapterConfig(r=4, targets=["c_attn"])], ids=["full", "lora"])
def test_interrupted_training_resumes_and_matches_an_uninterrupted_run(adapter: AdapterConfig | None, tiny_model: Path,
                                                                        tmp_path: Path) -> None:
    training = TrainingConfig(lr=1e-3, batch_size=4, max_length=64, resume_every=1, adapter=adapter)

    def run(workdir: Path, fail_at: int | None) -> list[dict[str, Any]]:
        loaded = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
        loaded.load("cpu")
        loaded.adapt(adapter)
        method = SFT(SFT.Config(), training)
        calls, original = {"n": 0}, method.loss

        def failing(backbone: Any, rows: Any) -> torch.Tensor:
            calls["n"] += 1
            if calls["n"] == fail_at:
                raise KeyboardInterrupt
            return original(backbone, rows)

        method.loss = failing  # type: ignore[method-assign]
        workdir.mkdir(exist_ok=True)
        fit(loaded, method, ROWS, training, workdir, 5, Progress())
        return [json.loads(line) for line in (workdir / "training.jsonl").read_text().splitlines()]

    whole = run(tmp_path / "whole", None)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "broken", 3)
    snapshot = torch.load(tmp_path / "broken" / "resume.pt", weights_only=True)["model"]
    assert snapshot and all(("lora_" in name) == (adapter is not None) for name in snapshot)  # only what trains
    resumed = run(tmp_path / "broken", None)
    assert [event["step"] for event in resumed] == [1, 2, 3] and not (tmp_path / "broken" / "resume.pt").exists()
    assert [event["loss"] for event in resumed] == pytest.approx([event["loss"] for event in whole], rel=1e-4)


def test_dpo_starts_at_log_two_and_caches_the_reference(backbone: TransformerBackbone, tmp_path: Path) -> None:
    rows = [json.loads(line) for line in Path("samples/llm_dpo.jsonl").read_text().splitlines()][:6]
    method = DPO(DPO.Config(beta=0.5), TrainingConfig(batch_size=4, max_length=64))
    prepared = method.prepare(backbone, rows, tmp_path, Progress())
    assert float(method.loss(backbone, prepared).detach()) == pytest.approx(math.log(2), abs=1e-4)  # policy == reference
    (tmp_path / "reference_margins.json").write_text(json.dumps([0.0] * 6))
    assert [row["_reference"] for row in method.prepare(backbone, rows, tmp_path, Progress())] == [0.0] * 6  # read, not recomputed


def test_last_token_pooling_handles_both_padding_sides() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    pooler = BiEncoder(BackboneConfig(architecture="bi_encoder", name=".", revision=None))
    right = pooler.pool(hidden, torch.tensor([[1, 1, 0], [1, 1, 1]]))
    assert right.tolist() == [hidden[0, 1].tolist(), hidden[1, 2].tolist()]
    left = pooler.pool(hidden, torch.tensor([[0, 1, 1], [1, 1, 1]]))
    assert left.tolist() == [hidden[0, 2].tolist(), hidden[1, 2].tolist()]


def test_rewards() -> None:
    row = {"answer": "12"}
    assert exact_match(" 12 ", row, {}) == 1.0 and exact_match("12 .", row, {}) == 0.0
    assert contains("It is 12 .", row, {}) == 1.0 and contains("13", row, {}) == 0.0
    worked = {"answer": "She sells 16 - 7 = 9 eggs for 9 * 2 = $18.\n#### 1,018"}
    assert last_number_match("So 9 * 2 gives 1018.", worked, {}) == 1.0
    assert last_number_match("The answer is 1018 eggs, not 9", worked, {}) == 0.0
    assert last_number_match("no digits here", worked, {}) == 0.0
    with pytest.raises(ValueError, match="no number"):
        last_number_match("3", {"answer": "unknown"}, {})


def test_resume_with_reused_samples_snapshots_between_groups_and_matches(tiny_model: Path, tmp_path: Path) -> None:
    """With `iterations` 2 a sampled group serves two steps. A snapshot inside the group would resume with fresh
    samples; so snapshots land on group boundaries only, and the resumed run repeats the uninterrupted one."""
    from modeling.llm.methods.grpo import GRPO

    rows = [json.loads(line) for line in Path("samples/llm_grpo.jsonl").read_text().splitlines()][:8]
    training = TrainingConfig(lr=1e-3, batch_size=4, max_length=64, resume_every=1, max_steps=4)
    snapshots: list[int] = []

    def run(workdir: Path, fail_at: int | None) -> list[dict[str, Any]]:
        loaded = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
        loaded.load("cpu")
        method = GRPO(GRPO.Config(group_size=2, max_new_tokens=3, reward={"name": "contains"}, iterations=2), training)
        calls, original = {"n": 0}, method.loss

        def failing(backbone: Any, batch: Any) -> torch.Tensor:
            calls["n"] += 1
            if calls["n"] == fail_at:
                raise KeyboardInterrupt
            return original(backbone, batch)

        method.loss = failing  # type: ignore[method-assign]
        workdir.mkdir(exist_ok=True)
        original_save = torch.save

        def spying_save(state: Any, path: Any) -> None:
            snapshots.append(int(state["step"]))
            original_save(state, path)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "save", spying_save)
            fit(loaded, method, rows, training, workdir, 5, Progress())
        return [json.loads(line) for line in (workdir / "training.jsonl").read_text().splitlines()]

    whole = run(tmp_path / "whole", None)
    assert snapshots == [2]  # resume_every=1, but step 1 and step 3 lie inside a group; step 4 is the last step
    snapshots.clear()
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "broken", 4)  # inside the second group, after the snapshot at step 2
    resumed = run(tmp_path / "broken", None)
    assert [event["step"] for event in resumed] == [1, 2, 3, 4]
    assert [event["loss"] for event in resumed] == pytest.approx([event["loss"] for event in whole], rel=1e-4)
    assert [event["reward"] for event in resumed] == [event["reward"] for event in whole]


def test_evaluation_runs_every_eval_every_steps_in_inference_mode(tiny_model: Path, tmp_path: Path) -> None:
    training = TrainingConfig(lr=1e-3, batch_size=4, max_length=64, epochs=2, max_steps=4, eval_every=2, resume_every=None)
    loaded = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(tiny_model)))
    loaded.load("cpu")
    method, modes = SFT(SFT.Config(), training), []

    def evaluation() -> dict[str, float]:
        modes.append(loaded.model.training)
        return {**method.evaluate(loaded, ROWS[:4], 4, Progress()), "count": 3}

    fit(loaded, method, ROWS, training, tmp_path, 5, Progress(), evaluation=evaluation)
    events = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert [event["step"] for event in events if "eval/loss" in event] == [2, 4] and modes == [False, False]
    assert events[1]["eval/count"] == 3.0 and "eval/loss" not in events[0]
    assert tracked(events[1]) == {**{f"train/{k}": v for k, v in events[1].items() if "/" not in k and k != "step"},
                                  **{k: v for k, v in events[1].items() if k.startswith("eval/")}}

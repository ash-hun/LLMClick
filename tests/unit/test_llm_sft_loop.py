"""`llm_sft` modeling/tuning/loop.py: the batch schedule, the learning rate, resuming an interrupted run, evaluation every N steps."""

import json
from typing import Any
from pathlib import Path

import torch
import pytest

from modeling.tuning.loop import fit, learning_rate, schedule, tracked
from modeling.tuning.config import AdapterConfig, TrainingConfig
from modeling.llm.models.transformer import TransformerBackbone
from modeling.llm.models.config import LLMBackboneConfig
from tests.recipe_inputs import SFT_ROWS as ROWS
from modeling.llm.methods.sft import SFT
from core.progress import Progress


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

"""`llm_grpo` end to end on a tiny random model: what the row check reads, sampled groups reused, and the KL term."""

import json
from typing import Any
from pathlib import Path

import pytest

from tests.recipe_inputs import raw
from core import pipeline


def test_changing_what_the_row_check_reads_checks_the_rows_again(tiny_model: Path, tmp_path: Path) -> None:
    exact = pipeline.build(raw("llm_grpo", tiny_model, tmp_path))
    config = raw("llm_grpo", tiny_model, tmp_path)
    config["method"]["reward"] = {"name": "last_number"}
    assert exact.fingerprint("data") != pipeline.build(config).fingerprint("data")
    config = raw("llm_grpo", tiny_model, tmp_path)
    config["method"]["group_size"] = 5  # not read by the row check: the data stage is shared
    assert exact.fingerprint("data") == pipeline.build(config).fingerprint("data")


def test_grpo_reuses_a_sampled_group_for_several_updates(tiny_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from modeling.llm.models.base import LLMBackbone

    sampled, original = {"n": 0}, LLMBackbone.generate

    def counting(self: Any, prompts: Any, max_new_tokens: int, samples: int = 1, *args: Any, **keys: Any) -> Any:
        sampled["n"] += samples > 1  # training samples groups; validation decodes one completion
        return original(self, prompts, max_new_tokens, samples, *args, **keys)

    monkeypatch.setattr(LLMBackbone, "generate", counting)
    config = raw("llm_grpo", tiny_model, tmp_path, max_steps=6)
    config["method"]["iterations"] = 3
    result = pipeline.build(config).run()
    log = [json.loads(line) for line in (Path(result["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert result["stages"]["train"]["steps"] == 6 and sampled["n"] == 2  # two groups, three updates each
    assert [event["reward"] for event in log[:3]] == [log[0]["reward"]] * 3  # the same group, so the same reward
    assert log[0]["ratio"] == pytest.approx(1.0) and log[3]["ratio"] == pytest.approx(1.0)  # first use of each group


def test_grpo_kl_needs_an_adapter_and_starts_at_zero(tiny_model: Path, tmp_path: Path) -> None:
    config = raw("llm_grpo", tiny_model, tmp_path)
    config["method"]["beta"] = 0.5
    with pytest.raises(ValueError, match="method.beta needs training.adapter"):
        pipeline.build(config)
    config["training"]["adapter"] = {"r": 4, "targets": ["c_attn"]}
    result = pipeline.build(config).run()
    log = [json.loads(line) for line in (Path(result["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert log[0]["kl"] == pytest.approx(0.0, abs=1e-6)  # an untrained adapter is the base model
    assert all(event["kl"] >= -1e-6 for event in log)

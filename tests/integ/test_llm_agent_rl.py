"""`llm_agent_rl` end to end on a tiny model: rollouts, the validate metrics, a rerun from cache, the environment's files."""

import json
from typing import Any
from pathlib import Path

import pytest

from core.progress_reporter import StateProgress
from core import pipeline_runner


def agent_config(model: Path, out: Path, **method: Any) -> dict[str, Any]:
    return {
        "pipeline": {"recipe": "llm_agent_rl", "name": "tiny", "seed": 3, "output_dir": str(out), "device": "cpu"},
        "model": {"architecture": "transformer", "name": str(model)},
        "data": {"sources": [{"name": "local_jsonl", "path": "samples/llm_agent_rl.jsonl", "limit": 20}], "validation": 0.2},
        "method": {"group_size": 2, "max_turns": 2, "max_new_tokens": 6, "reward": {"name": "exact_match"}, **method},
        "training": {"lr": 1e-3, "batch_size": 2, "max_length": 512, "max_steps": 2},
    }


def test_agent_rl_trains_validates_and_reruns_from_cache(bpe_model: Path, tmp_path: Path) -> None:
    first = pipeline_runner.build(agent_config(bpe_model, tmp_path)).run()
    metrics = first["stages"]["validate"]["metrics"]
    assert set(metrics) >= {"reward", "turns", "tool_calls", "malformed", "answered"} and 1 <= metrics["turns"] <= 2
    log = [json.loads(line) for line in (Path(first["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert len(log) == 2 and "turns" in log[0]
    progress = StateProgress()
    assert json.dumps(pipeline_runner.build(agent_config(bpe_model, tmp_path), progress).run()) == json.dumps(first)


def test_environment_rules_and_files(bpe_model: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        pipeline_runner.build(agent_config(bpe_model, tmp_path, environment={"name": "shell"}))
    docs = tmp_path / "docs.jsonl"
    docs.write_text(json.dumps({"title": "a", "text": "b"}) + "\n")
    config = agent_config(bpe_model, tmp_path, environment={"name": "lookup", "path": str(docs)})
    before = pipeline_runner.build(config).fingerprint("train")
    assert str(docs) in pipeline_runner.build(config).config.paths()
    docs.write_text(json.dumps({"title": "a", "text": "c"}) + "\n")
    assert pipeline_runner.build(config).fingerprint("train") != before

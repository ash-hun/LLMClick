"""`llm_pretrain` end to end on a tiny byte-level BPE model: packed Korean text, alone and after extending the tokenizer."""

import json
from typing import Any
from pathlib import Path

from core.progress_reporter import StateProgress
from core import pipeline_runner


def pretrain_config(model: Path, out: Path, **training: Any) -> dict[str, Any]:
    return {
        "pipeline": {"recipe": "llm_pretrain", "name": "tiny", "seed": 3, "output_dir": str(out), "device": "cpu"},
        "model": {"architecture": "transformer", "name": str(model)},
        "data": {"sources": [{"name": "local_jsonl", "path": "samples/llm_pretrain_ko.jsonl"}], "validation": 0.2},
        "training": {"lr": 3e-3, "batch_size": 4, "max_length": 64, **training},
        "validation": {"max": {"nll_per_char": 10.0}},
    }


def test_pretraining_lowers_the_loss_and_a_rerun_builds_nothing(bpe_model: Path, tmp_path: Path) -> None:
    first = pipeline_runner.build(pretrain_config(bpe_model, tmp_path, epochs=3)).run()
    log = [json.loads(line) for line in (Path(first["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    assert log[-1]["loss"] < log[0]["loss"] * 0.6 and first["stages"]["data"]["rows"] == {"train": 96, "validation": 24}
    assert set(first["stages"]["validate"]["metrics"]) >= {"loss", "perplexity", "nll_per_char", "tokens"}
    progress = StateProgress()
    assert json.dumps(pipeline_runner.build(pretrain_config(bpe_model, tmp_path, epochs=3), progress).run()) == json.dumps(first)
    assert progress.snapshot()["stages"] == {"data": "cached", "train": "cached", "validate": "cached"}


def test_new_tokens_learn_under_lora_and_the_text_costs_fewer_tokens(bpe_model: Path, tmp_path: Path) -> None:
    words = ["토크나이저에", "체크포인트는", "파이프라인은", "모델을", "데이터는"]
    plain = pipeline_runner.build(pretrain_config(bpe_model, tmp_path / "plain", max_steps=6)).run()
    config = pretrain_config(bpe_model, tmp_path / "extended", max_steps=6, adapter={"r": 4, "targets": ["c_attn"]})
    config["model"]["tokenizer"] = {"tokens": words}
    extended = pipeline_runner.build(config).run()
    assert extended["stages"]["validate"]["metrics"]["tokens"] < plain["stages"]["validate"]["metrics"]["tokens"]

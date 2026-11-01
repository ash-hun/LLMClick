"""`llm_quantize` end to end with the scheme replaced by a copy: base loss measured, delta reported, rerun cached."""

from typing import Any
from pathlib import Path

import pytest

from modeling.quantize.quantize_scheme import SCHEMES, QuantizeJob
from core.progress_reporter import StateProgress
from tests.recipe_inputs import raw
from core import pipeline_runner


def quantize_raw(model: Path, out: Path, **quantization: Any) -> dict[str, Any]:
    config = raw("llm_sft", model, out)
    config["pipeline"]["recipe"] = "llm_quantize"
    config["training"] = {"max_length": 64}
    config["quantization"] = {"name": "gptq", "rows": 8, **quantization}
    config["validation"] = {"max": {"loss_delta": 0.01}}
    return config


@pytest.fixture
def copied(monkeypatch: pytest.MonkeyPatch) -> list[QuantizeJob]:
    jobs: list[QuantizeJob] = []

    def copy(job: QuantizeJob) -> None:
        jobs.append(job)
        job.loader.from_pretrained(job.origin).save_pretrained(job.out)
        job.tokenizer.save_pretrained(job.out)

    monkeypatch.setitem(SCHEMES._items, "gptq", copy)
    return jobs


def test_quantize_measures_against_the_base_and_reruns_from_cache(tiny_model: Path, tmp_path: Path, copied: list[QuantizeJob]) -> None:
    first = pipeline_runner.build(quantize_raw(tiny_model, tmp_path, scheme="W8A16")).run()
    job = copied[0]
    assert len(job.calibration) == 8 and job.ignore == ["lm_head"] and job.params == {"scheme": "W8A16"}
    metrics = first["stages"]["validate"]["metrics"]
    assert metrics["loss_delta"] == pytest.approx(0.0, abs=1e-5) and metrics["size_mb"] > 0
    assert metrics["base_loss"] == pytest.approx(metrics["loss"], abs=1e-5)
    progress = StateProgress()
    second = pipeline_runner.build(quantize_raw(tiny_model, tmp_path, scheme="W8A16"), progress).run()
    assert second == first and len(copied) == 1
    assert progress.snapshot()["stages"] == {"data": "cached", "train": "cached", "validate": "cached"}


def test_quantize_config_rules(tiny_model: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown quantization scheme 'int3'"):
        pipeline_runner.build(quantize_raw(tiny_model, tmp_path, name="int3"))
    config = quantize_raw(tiny_model, tmp_path)
    config["training"]["adapter"] = {"targets": ["c_attn"]}
    with pytest.raises(ValueError, match="trains nothing"):
        pipeline_runner.build(config)
    a = pipeline_runner.build(quantize_raw(tiny_model, tmp_path))
    b = pipeline_runner.build(quantize_raw(tiny_model, tmp_path, name="awq"))
    assert a.fingerprint("data") == b.fingerprint("data") and a.fingerprint("train") != b.fingerprint("train")

"""`llm_distill` end to end on tiny models: a trained teacher, its files in the fingerprint, the student moving towards it."""

import json
import shutil
from typing import Any
from pathlib import Path

from tests.recipe_inputs import raw
from core import pipeline_runner


def with_teacher(tiny_model: Path, out: Path, teacher: Path, **training: Any) -> dict[str, Any]:
    config = raw("llm_distill", tiny_model, out, **training)
    config["method"]["teacher"]["name"] = str(teacher)
    return config


def test_student_closes_the_gap_to_a_trained_teacher(tiny_model: Path, tmp_path: Path) -> None:
    sft = pipeline_runner.build(raw("llm_sft", tiny_model, tmp_path / "teacher", max_steps=40, lr=3e-3)).run()
    teacher = Path(sft["stages"]["train"]["checkpoint"])
    config = with_teacher(tiny_model, tmp_path, teacher, max_steps=16, lr=3e-3)
    config["method"].update({"samples": 4, "max_new_tokens": 4})
    result = pipeline_runner.build(config).run()
    log = [json.loads(line) for line in (Path(result["stages"]["train"]["run"]) / "training.jsonl").read_text().splitlines()]
    first, last = sum(e["reverse_kl"] for e in log[:4]) / 4, sum(e["reverse_kl"] for e in log[-4:]) / 4
    assert first > 0 and last < first - 0.05
    assert set(result["stages"]["validate"]["metrics"]) >= {"teacher_logp", "reverse_kl"}


def test_a_local_teacher_counts_by_content(tiny_model: Path, tmp_path: Path) -> None:
    teacher = tmp_path / "teacher"
    shutil.copytree(tiny_model, teacher)

    def fingerprints() -> tuple[str, str]:
        pipeline = pipeline_runner.build(with_teacher(tiny_model, tmp_path, teacher))
        return pipeline.fingerprint("data"), pipeline.fingerprint("train")

    data, train = fingerprints()
    (teacher / "config.json").write_text((teacher / "config.json").read_text() + "\n")
    assert fingerprints() == (data, fingerprints()[1]) and fingerprints()[1] != train


def test_teacher_path_is_confined_like_the_model(tiny_model: Path, tmp_path: Path) -> None:
    config = with_teacher(tiny_model, tmp_path, Path("../elsewhere"))
    config["method"]["teacher"]["revision"] = "a" * 40
    assert "../elsewhere" in pipeline_runner.build(config).config.paths()

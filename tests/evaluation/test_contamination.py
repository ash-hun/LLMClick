"""Overlap between training rows and evaluation items: exact matches, near-duplicates by n-grams, and its reporting."""

import json
from pathlib import Path
from typing import Any


from core import pipeline
from evaluation.contamination import Overlap, measure, shingles, texts_of
from tests.evaluation.test_custom import evaluation, trained  # noqa: F401


def test_texts_and_shingles() -> None:
    row = {"messages": [{"role": "user", "content": "What is 2 plus 3 ?"}, {"role": "assistant", "content": "5"}], "_chain": "hidden", "n": 3}
    assert texts_of(row) == ["user", "What is 2 plus 3 ?", "assistant", "5"]
    words = " ".join(f"w{i}" for i in range(10))
    assert len(shingles(words, 8)) == 3 and len(shingles("short text", 8)) == 1 and shingles("", 8) == []


def test_exact_and_near_duplicates_are_counted() -> None:
    training = ["the quick brown fox jumps over the lazy dog near the river bank today", "an unrelated sentence about trains"]
    overlap = Overlap(training)
    assert overlap.is_exact("The quick  brown fox jumps over the lazy dog near the river bank today")
    assert overlap.share("the quick brown fox jumps over the lazy dog near the river bank tomorrow") > 0.5
    assert overlap.share("completely different words that never appeared anywhere in training rows") == 0.0


def test_measure_reads_the_training_rows(tmp_path: Path) -> None:
    rows = [{"messages": [{"role": "user", "content": f"question number {i} about the weather in the mountains of the north"},
                          {"role": "assistant", "content": f"answer {i}"}]} for i in range(5)]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    items = [rows[0], {"messages": [{"role": "user", "content": "question number 0 about the weather in the mountains of the north plus more"}]},
             {"messages": [{"role": "user", "content": "nothing like it at all, truly"}]}]
    result = measure(path, items)
    assert result is not None and result["training_rows"] == 5 and result["items"] == 3
    assert result["exact"] == 1 and result["near"] == 2 and result["flagged"] == [0, 1] and result["threshold"] == 0.5
    assert measure(tmp_path / "missing.jsonl", items) is None and measure(None, items) is None


def test_base_checkpoints_report_no_contamination(trained: dict[str, Any], tmp_path: Path) -> None:  # noqa: F811
    body = evaluation(trained["directory"], trained["stages"]["data"]["validation"], tmp_path, "base")
    result = pipeline.build(body).run()
    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    assert report["contamination"] is None  # the base weights never saw the experiment's rows
    contaminated = evaluation(trained["directory"], trained["stages"]["data"]["train"], tmp_path)  # the rows it trained on
    report = json.loads(Path(pipeline.build(contaminated).run()["stages"]["report"]["report"]).read_text())
    assert report["contamination"]["exact"] == 64 and report["contamination"]["near"] == 64

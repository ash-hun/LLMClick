"""`evaluation_decision`: a decision checkpoint on Jev rows and JevBench tiers, without and with reasoning."""

import json
from pathlib import Path
from typing import Any

import pytest

from core import pipeline
from core.progress import StateProgress
from evaluation.decision import jevbench, summarize
from tests.modeling.tuning.test_decision import raw as decision_raw


@pytest.fixture(scope="module")
def decision(tiny_model: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("decision")
    return pipeline.build(decision_raw("llm_decision_sft", tiny_model, out, "readout")).run()


def config(experiment: str, rows: dict[str, Any], out: Path, **keys: Any) -> dict[str, Any]:
    return {"pipeline": {"recipe": "evaluation_decision", "name": "dec", "output_dir": str(out), "device": "cpu"},
            "model": {"experiment": experiment, "checkpoint": "validate"}, "data": {"sources": [rows]},
            "evaluation": {"batch_size": 4}, **keys}


def test_scores_without_and_with_reasoning(decision: dict[str, Any], tmp_path: Path) -> None:
    rows = {"name": "local_jsonl", "path": decision["stages"]["data"]["validation"]}
    progress = StateProgress()
    result = pipeline.build(config(decision["directory"], rows, tmp_path, max_think=4), progress).run()
    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    assert set(report["scores"]) == {"nothink", "think"} and report["settings"]["max_think"] == 4 and report["settings"]["think"] == "both"
    plain, think = report["scores"]["nothink"], report["scores"]["think"]
    assert plain["accuracy"] == pytest.approx(decision["stages"]["validate"]["metrics"]["accuracy"])  # the validate stage's number
    assert plain["ece"] == pytest.approx(decision["stages"]["validate"]["metrics"]["ece"], abs=1e-6)
    assert plain["think_tokens"] == 0 and 0 < think["think_tokens"] <= 4 and 0 <= think["closed"] <= 1
    assert plain["seconds_p50"] > 0 and plain["seconds_p95"] >= plain["seconds_p50"] and "by_tier" not in plain
    items = [json.loads(line) for line in Path(report["items"]).read_text().splitlines()]
    assert len(items) == 2 * plain["questions"] and {item["mode"] for item in items} == {"nothink", "think"}
    assert all(0 <= item["confidence"] <= 1 and item["options"] >= 2 for item in items)
    again = pipeline.build(config(decision["directory"], rows, tmp_path, max_think=4), StateProgress()).run()
    assert json.dumps(again) == json.dumps(result)
    only = pipeline.build(config(decision["directory"], rows, tmp_path, think="off")).run()
    assert list(only["stages"]["report"]["scores"]) == ["nothink"]


def test_jevbench_tiers_are_read_and_reported_with_their_chance_levels(decision: dict[str, Any], tmp_path: Path) -> None:
    bench = tmp_path / "jevbench"
    bench.mkdir()
    state = "Order total: $62. Free shipping applies to orders of $70 or more."
    (bench / "easy.jsonl").write_text(json.dumps({
        "id": "e1", "family": "tiers", "state": state, "labels": ["bronze", "silver", "gold"], "expected": "silver",
        "question": {"type": "choice", "instructions": "Which tier?", "criteria": {"bronze": "under $30", "silver": "$30 to $69", "gold": "$70 or more", "platinum": "unused"}}}) + "\n")
    (bench / "original.jsonl").write_text(json.dumps({
        "id": "s1", "family": "shipping", "state": state, "labels": ["no", "yes"], "expected": "no",
        "question": {"type": "noul", "instructions": "Does free shipping apply?", "criteria": None}}) + "\n")
    (bench / "hard.jsonl").write_text(json.dumps({
        "id": "h1", "family": "lateness", "state": state, "labels": ["0", "1", "2"], "expected": "2",
        "question": {"type": "score", "instructions": "How late?", "criteria": ["On time", "A bit", "Very"]}}) + "\n")
    rows = jevbench({"path": str(bench)})
    assert [(row["tier"], row["id"], list(row["questions"])) for row in rows] == [("easy", "e1", ["decision"]), ("standard", "s1", ["decision"]), ("hard", "h1", ["decision"])]
    assert list(rows[0]["questions"]["decision"]["criteria"]) == ["bronze", "silver", "gold"]  # only the listed labels
    assert rows[1]["questions"]["decision"]["label"] is False and rows[2]["questions"]["decision"]["label"] == 2
    assert [row["tier"] for row in jevbench({"path": str(bench), "tiers": ["hard"]})] == ["hard"]
    with pytest.raises(ValueError, match="unknown JevBench tier"):
        jevbench({"path": str(bench), "tiers": ["judge"]})

    result = pipeline.build(config(decision["directory"], {"name": "jevbench", "path": str(bench)}, tmp_path, think="off")).run()
    scores = result["stages"]["report"]["scores"]["nothink"]
    assert scores["questions"] == 3 and set(scores["by_tier"]) == {"easy", "standard", "hard"}
    assert scores["by_tier"]["easy"]["chance"] == pytest.approx(1 / 3) and scores["by_tier"]["standard"]["chance"] == 0.5
    assert 0 <= scores["corrected_mean"] <= 1


def test_summary_arithmetic() -> None:
    records = [{"correct": True, "p_label": 0.8, "confidence": 0.8, "think_tokens": 2, "closed": True, "seconds": 0.1, "tier": "easy", "options": 2},
               {"correct": False, "p_label": 0.25, "confidence": 0.75, "think_tokens": 4, "closed": False, "seconds": 0.3, "tier": "easy", "options": 4}]
    summary = summarize(records)
    assert summary["accuracy"] == 0.5 and summary["think_tokens"] == 3 and summary["closed"] == 0.5
    assert summary["seconds_p50"] == 0.3 and summary["by_tier"]["easy"]["chance"] == pytest.approx(0.375)
    assert summary["by_tier"]["easy"]["corrected"] == pytest.approx((0.5 - 0.375) / 0.625)
    assert summarize([]) == {}


def test_a_non_decision_experiment_is_refused(tiny_model: Path, tmp_path: Path) -> None:
    from tests.modeling.tuning.test_recipes import raw

    sft = pipeline.build(raw("llm_sft", tiny_model, tmp_path)).run()
    rows = {"name": "local_jsonl", "path": "samples/llm_decision.jsonl"}
    with pytest.raises(ValueError, match="does not fit|not a decision experiment"):
        pipeline.build(config(sft["directory"], rows, tmp_path)).run()

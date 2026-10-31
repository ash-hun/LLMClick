"""`evaluation_decision` evaluation/decision.py: the summary arithmetic, and JevBench tiers downloaded once into the cache."""

import json
from typing import Any
from pathlib import Path

import pytest

from evaluation.decision import jevbench, summarize


def test_summary_arithmetic() -> None:
    records = [{"correct": True, "p_label": 0.8, "confidence": 0.8, "think_tokens": 2, "closed": True, "seconds": 0.1, "tier": "easy", "options": 2},
               {"correct": False, "p_label": 0.25, "confidence": 0.75, "think_tokens": 4, "closed": False, "seconds": 0.3, "tier": "easy", "options": 4}]
    summary = summarize(records)
    assert summary["accuracy"] == 0.5 and summary["think_tokens"] == 3 and summary["closed"] == 0.5
    assert summary["seconds_p50"] == 0.3 and summary["by_tier"]["easy"]["chance"] == pytest.approx(0.375)
    assert summary["by_tier"]["easy"]["corrected"] == pytest.approx((0.5 - 0.375) / 0.625)
    assert summarize([]) == {}


def test_jevbench_tiers_are_downloaded_once_into_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from evaluation import decision

    served = {"easy.jsonl": json.dumps({"id": "e", "family": "f", "state": "s", "labels": ["a", "b"], "expected": "a",
                                        "question": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}}) + "\n",
              "original.jsonl": "", "hard.jsonl": ""}
    requested: list[str] = []

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

    def fake_urlopen(url: str) -> Response:
        requested.append(url)
        return Response(served[url.rsplit("/", 1)[1]].encode())

    monkeypatch.setattr(decision, "urlopen", fake_urlopen)
    monkeypatch.setattr(decision, "CACHE", tmp_path / "cache")
    rows = jevbench({"revision": "abc123"})
    assert [row["id"] for row in rows] == ["e"] and len(requested) == 3
    assert requested[0] == "https://raw.githubusercontent.com/fstandhartinger/jevbench/abc123/datasets/public/easy.jsonl"
    assert jevbench({"revision": "abc123", "tiers": ["easy"]}) == rows and len(requested) == 3  # cached: nothing fetched again
    assert (tmp_path / "cache" / "jevbench" / "abc123" / "hard.jsonl").is_file()

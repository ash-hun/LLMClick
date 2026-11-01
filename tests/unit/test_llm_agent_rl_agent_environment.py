"""`llm_agent_rl` modeling/llm/agent/agent_environment.py: the calculator parses instead of running, lookup ranks documents."""

import json
from pathlib import Path

import pytest

from modeling.llm.agent.agent_environment import ENVIRONMENTS


def test_the_calculator_does_arithmetic_and_nothing_else() -> None:
    calculator = ENVIRONMENTS.get("calculator")({})
    assert calculator.call("calculate", {"expression": "(1,200 + 7) * 3"}) == "3621"
    assert calculator.call("calculate", {"expression": "7 / 2"}) == "3.5"
    for expression in ("__import__('os').system('ls')", "2 ** 1000", "x + 1", "1 / 0", "9" * 300):
        assert calculator.call("calculate", {"expression": expression}).startswith("error:")
    assert calculator.call("delete_files", {}).startswith("error: no tool named")
    assert calculator.call("calculate", {"wrong": "1"}).startswith("error:")


def test_lookup_returns_the_documents_sharing_most_words(tmp_path: Path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text("".join(json.dumps(doc) + "\n" for doc in [{"title": "Seoul", "text": "capital of Korea"},
                                                             {"title": "Busan", "text": "port city of Korea"},
                                                             {"title": "Lima", "text": "capital of Peru"}]))
    lookup = ENVIRONMENTS.get("lookup")({"path": str(path), "top": 1})
    assert json.loads(lookup.call("search", {"query": "port city"}))[0]["title"] == "Busan"
    with pytest.raises(ValueError, match="needs `path`"):
        ENVIRONMENTS.get("lookup")({})

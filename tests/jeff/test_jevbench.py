import pytest

from jeff import jevbench
from jeff.data import validate


def raw(kind: str, expected: object, criteria: object = None) -> dict:
    return {"id": "hard-x-01", "group": "hard-x", "family": "trap", "expected": expected, "state": "Facts.",
            "question": {"type": kind, "instructions": "Decide.", "criteria": criteria}}


def test_choice_and_yes_no_items_become_evaluation_rows() -> None:
    choice = jevbench.convert(raw("choice", "deny", {"deny": "Deny.", "grant": "Grant."}))
    noul = jevbench.convert(raw("noul", "no", {"true": "Yes.", "false": "No."}))
    assert choice is not None and choice["label"] == "deny" and choice["suite"] == "JevBench public hard"
    assert noul is not None and noul["label"] is False
    validate([choice])
    validate([noul | {"id": "other"}])


def test_score_items_are_left_out_and_bad_answers_fail_loudly() -> None:
    assert jevbench.convert(raw("score", 1, ["none", "one"])) is None
    with pytest.raises(ValueError, match="not an option"):
        jevbench.convert(raw("choice", "maybe", {"deny": "Deny."}))
    with pytest.raises(ValueError, match="expects"):
        jevbench.convert(raw("noul", "perhaps"))

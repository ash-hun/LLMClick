from collections import Counter

from jeff import escape
from jeff.data import validate

CRITERIA = {"billing": "Charges, invoices or refunds.", "technical": "Something is broken.", "delivery": "Shipping or missing parcels."}


def row(identifier: str, state: object = "My parcel never arrived.", criteria: dict = CRITERIA, label: str = "delivery") -> dict:
    return {"id": identifier, "suite": "s", "family": identifier, "state": state,
            "question": {"type": "choice", "instructions": "Route it.", "criteria": dict(criteria)},
            "label": label, "target": label, "source": {"dataset": "s"}}


def test_the_escape_option_is_the_answer_exactly_when_the_right_option_was_removed() -> None:
    made = [escape.add_escape(row(f"r{i}"), seed=1) for i in range(400)]
    for r in made:
        keys = list(r["question"]["criteria"])
        added = r["source"]["escape_option"]
        assert added in keys and r["question"]["criteria"][added].lower() == dict(escape.WORDINGS)[added].lower()
        if r["source"]["escape_is_answer"]:
            assert r["label"] == added and "delivery" not in keys
        else:
            assert r["label"] == "delivery" and "delivery" in keys
    assert 150 < sum(r["source"]["escape_is_answer"] for r in made) < 250
    assert len(Counter(r["source"]["escape_option"] for r in made)) == len(escape.WORDINGS)  # every wording is used
    shown = [r["question"]["criteria"][r["source"]["escape_option"]] for r in made]
    assert any(t.isupper() for t in shown) and any(t.islower() for t in shown)  # capitalisation varies
    assert len({list(r["question"]["criteria"]).index(r["source"]["escape_option"]) for r in made}) >= 3  # varied positions
    validate(made)


def test_only_sound_rows_are_eligible() -> None:
    assert escape.eligible(row("ok"))
    assert not escape.eligible(row("two", criteria={"yes": "Yes, it is.", "no": "No, it is not."}, label="yes"))
    assert not escape.eligible(row("generic", criteria={"A": "Option A.", "B": "Option B.", "C": "Option C."}, label="A"))
    assert not escape.eligible(row("listed", state="Pick one.\nbilling: charges\ntechnical: broken\ndelivery: parcels"))
    assert not escape.eligible(row("has_other", criteria={**CRITERIA, "other": "Something else."}))
    soft = row("soft")
    soft["target"] = [0.2, 0.3, 0.5]
    assert not escape.eligible(soft)
    rows, changed = escape.add([row(f"x{i}") for i in range(2000)], seed=4)
    assert len(rows) == 2000 and 60 <= changed <= 140

import random
from fractions import Fraction

from jeff import probability, puzzles
from jeff.data import validate


def test_validity_checker_on_classic_arguments() -> None:
    barbara = [("all", 0, 1), ("all", 1, 2)]
    assert puzzles.valid(barbara, ("all", 0, 2), 3, []) is True  # every A is B, every B is C: every A is C
    middle = [("all", 0, 1), ("all", 2, 1)]
    assert puzzles.valid(middle, ("all", 0, 2), 3, []) is False  # undistributed middle
    tollens = [("all", 0, 1), ("person", "Lisa", 1, False)]
    assert puzzles.valid(tollens, ("person", "Lisa", 0, False), 3, ["Lisa"]) is True  # modus tollens
    affirming = [("all", 0, 1), ("person", "Lisa", 1, True)]
    assert puzzles.valid(affirming, ("person", "Lisa", 0, True), 3, ["Lisa"]) is False  # affirming the consequent
    contradiction = [("person", "Lisa", 0, True), ("person", "Lisa", 0, False)]
    assert puzzles.valid(contradiction, ("all", 0, 1), 3, ["Lisa"]) is None


def test_formal_fallacies_rows_for_both_verdicts() -> None:
    for index in range(20):
        label = ("valid", "invalid")[index % 2]
        text = puzzles.formal_fallacies({"id": f"syn-1-formal_fallacies-{index:06d}", "family": "formal_fallacies",
                                          "label": label, "difficulty": "hard"})
        assert text.startswith('"') and "It follows that" in text and text.count("First,") == 1
        assert "every a " not in text and "no a " not in text and "every an " not in text


def test_probabilities_are_exact_and_answers_balanced() -> None:
    text, question, p = probability.cards(random.Random(1))
    drawn = int(text.split(" cards are dealt")[0].split()[-1])
    from math import comb
    assert p == 1 - Fraction(comb(48, drawn), comb(52, drawn))
    rows = probability.build(400, seed=5)
    assert len({(r["state"], r["question"]["instructions"]) for r in rows}) == 400  # no repeated question
    assert sum(r["label"] is True for r in rows) == 200
    assert all(abs(r["target"] - r["source"]["probability"]) < 1e-4 and (r["target"] > 0.5) == r["label"] for r in rows)
    assert all(abs(r["source"]["probability"] - 0.5) >= 0.03 for r in rows)
    validate(rows)

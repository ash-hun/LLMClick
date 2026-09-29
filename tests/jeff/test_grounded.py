import random

from jeff import grounded


def slot(label: str) -> dict:
    return {"id": "s1", "family": "grounded_pairwise", "label": label, "question": "Q?", "correct_answer": "42", "wrong_answer": "41"}


def text(a: str, b: str) -> str:
    return f"Question: Q?\nResponse A: first I add.\nFinal answer: {a}\nResponse B: then I add.\nFinal answer: {b}"


def test_matching_numbers_and_option_texts() -> None:
    assert grounded.same_answer("$42", "42") and grounded.same_answer("42.0 apples", "42")
    assert not grounded.same_answer("41", "42")
    assert grounded.same_answer("C) a supermarket", "C) a supermarket")
    assert not grounded.same_answer("B) an oven", "C) a supermarket")


def test_problems_empty_when_answers_sit_where_the_label_says() -> None:
    assert grounded.problems(slot("A"), text("42", "41")) == []
    assert grounded.problems(slot("B"), text("41", "42")) == []
    assert grounded.problems(slot("A"), text("41", "42")) == ["Response A must end with 'Final answer: 42'",
                                                              "Response B must end with 'Final answer: 41'"]
    assert grounded.problems(slot("A"), "Question: Q?\nResponse A: x\nResponse B: y") == [
        "Response A has no 'Final answer:' line", "Response B has no 'Final answer:' line"]


def test_wrong_numbers_are_plausible_and_different() -> None:
    for answer in (0, 7, 18, 250, -3):
        rng = random.Random(answer)
        wrong = grounded.wrong_number(answer, rng)
        assert wrong != answer and (wrong >= 0 or answer < 0)


def test_dress_only_touches_grounded_slots() -> None:
    items = [{"id": "g", "question": "Q?", "correct_answer": "1", "wrong_answer": "2"}]
    slots = [{"id": "a", "family": "grounded_pairwise"}, {"id": "b", "family": "sarcasm"}]
    out = grounded.dress(slots, items, seed=1)
    assert out[0]["correct_answer"] == "1" and "question" not in out[1]


def test_long_pairwise_draws_only_arguable_questions() -> None:
    items = [{"id": f"extra-commonsense_qa-{i}", "question": "q", "correct_answer": "a", "wrong_answer": "b"} for i in range(50)]
    items.append({"id": "gsm8k-7", "question": "2+2?", "correct_answer": "4", "wrong_answer": "5"})
    slots = [{"id": f"s{i}", "family": family, "label": "A"} for i in range(20) for family in ("long_pairwise", "grounded_pairwise")]
    dressed = grounded.dress(slots, items, seed=1)  # type: ignore[arg-type]
    assert {s["question"] for s in dressed if s["family"] == "long_pairwise"} == {"2+2?"}
    assert any(s["question"] == "q" for s in dressed if s["family"] == "grounded_pairwise")

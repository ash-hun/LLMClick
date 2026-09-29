import asyncio
from fractions import Fraction

from jeff import sampled
from jeff.data import validate


def test_reference_answers_and_final_lines_become_exact_numbers() -> None:
    assert sampled.boxed(r"So the answer is $\boxed{\frac{3}{4}}$.") == r"\frac{3}{4}"
    assert sampled.boxed(r"First \boxed{2}, then \boxed{\dfrac{10}{4}}") == r"\dfrac{10}{4}"
    assert sampled.as_number(r"\dfrac{10}{4}") == Fraction(5, 2) == sampled.as_number("2.5") == sampled.as_number("5/2")
    assert sampled.as_number("-12") == -12 and sampled.as_number("1,000") == 1000
    assert sampled.as_number(r"\sqrt{2}") is None and sampled.as_number("x+1") is None and sampled.as_number("3/0") is None
    assert sampled.final_answer("Work.\nFinal answer: 7/4") == Fraction(7, 4)
    assert sampled.final_answer("No final line here.") is None


class Teacher:
    def __init__(self, answers: dict[int, str]) -> None:
        self.answers = answers

    async def complete(self, *, system, user, schema, temperature, max_tokens, seed):
        return {"solution": f"Reasoning for attempt {seed}.\nFinal answer: {self.answers[seed]}"}


def test_a_problem_gives_a_pair_only_when_one_attempt_is_right_and_another_wrong() -> None:
    problem = {"id": "math-algebra-1", "problem": "What is 3/4 + 1/4?", "answer": Fraction(1), "level": "Level 3", "subject": "algebra"}
    mixed = asyncio.run(sampled.process(Teacher({0: "1", 1: "2", 2: "1.0", 3: "7/4"}), problem))  # type: ignore[arg-type]
    row = mixed["row"]
    assert mixed["right"] == 2 and mixed["wrong"] == 2 and row is not None
    correct = row["state"].split(f"Response {row['label']}: ", 1)[1]
    assert correct.startswith("Reasoning for attempt 0.")  # the right attempt sits where the label says
    validate([row])
    all_right = asyncio.run(sampled.process(Teacher({s: "1" for s in range(4)}), problem))  # type: ignore[arg-type]
    assert all_right["row"] is None

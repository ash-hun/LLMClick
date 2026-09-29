from jeff import prm
from jeff.data import validate


def record(problem: str, truth: str, answer: str, steps: list[str]) -> dict:
    return {"question": {"problem": problem, "ground_truth_answer": truth, "pre_generated_answer": answer,
                         "pre_generated_steps": repr(steps)}}


def test_pairs_put_a_right_and_a_wrong_solution_together() -> None:
    records = [record("What is 1/2 + 1/4?", r"\frac{3}{4}", "0.75", ["Add the fractions.", "# Answer\n\n0.75"]),
               record("What is 1/2 + 1/4?", r"\frac{3}{4}", "2/6", ["Add tops and bottoms.", "# Answer\n\n2/6"]),
               record("What is 2+2?", "4", "4", ["Four.", "# Answer\n\n4"]),
               {"question": {"problem": "What is 2+2?", "ground_truth_answer": "4", "pre_generated_answer": "5",
                             "pre_generated_steps": ["Five, surely.", "# Answer\n\n5"]}},  # steps stored as a list  # only right answers: no pair
               record("Name the shape.", "square", "circle", ["It is round."])]  # not a number: not used
    rows = prm.pairs(records, seed=1)
    assert len(rows) == 2
    row = next(r for r in rows if "1/2 + 1/4" in r["state"])
    right = row["state"].split(f"Response {row['label']}: ", 1)[1]
    assert right.startswith("Add the fractions.")
    validate(rows)

from jeff import codecontests
from jeff.data import validate


def test_an_accepted_and_a_rejected_python3_program_make_a_pair() -> None:
    problem = {"name": "1A. Theatre Square", "description": "Cover the square with flagstones.",
               "solutions": {"language": [2, 3], "solution": ["int main(){}", "print(4)"]},
               "incorrect_solutions": {"language": [3], "solution": ["print(5)"]}}
    row = codecontests.pair(problem, seed=1)
    assert row is not None
    accepted = row["state"].split(f"Response {row['label']}: ", 1)[1]
    assert accepted.startswith("```python\nprint(4)")  # the C++ solution is ignored; the accepted one sits at the label
    assert row["state"].startswith("Question: Write a Python 3 program that solves this problem.")
    validate([row])
    no_wrong = {**problem, "incorrect_solutions": {"language": [2], "solution": ["int main(){return 1;}"]}}
    assert codecontests.pair(no_wrong, seed=1) is None

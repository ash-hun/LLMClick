"""JudgeBench-style pairs from PRM800K: two real model solutions to the same maths problem, one right and one wrong.

PRM800K (OpenAI, MIT licence; rows on Hugging Face as tasksource/PRM800K) holds about 98,000 step-by-step solutions by
a strong model to MATH problems, rated by people. For every problem whose reference answer is a plain number, a
solution whose own final answer equals it and one whose final answer is a different number become responses A and B
(which is which follows a hash of the problem), asking which answers the question more correctly - JudgeBench's own
recipe. Solutions whose final answer cannot be read as a number are not used."""

import argparse
import ast
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from jeff.data import validate, write_rows
from jeff.sampled import QUESTION, as_number
from jeff.types import Example

SOURCE = ("tasksource/PRM800K", "547b19506677a59037ee888838834b65e9b1ddd4", "phase2_train.jsonl")


def solution_text(record: dict) -> str:
    raw = record["question"]["pre_generated_steps"]
    steps = raw if isinstance(raw, list) else ast.literal_eval(raw)  # the file stores some as lists, some as their text
    return "\n".join(str(step).strip() for step in steps if str(step).strip())


def pairs(records: list[dict], seed: int) -> list[Example]:
    by_problem: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"right": [], "wrong": []})
    for record in records:
        question = record["question"]
        truth, given = as_number(question["ground_truth_answer"] or ""), as_number(question["pre_generated_answer"] or "")
        if truth is None or given is None:
            continue
        by_problem[question["problem"]]["right" if given == truth else "wrong"].append(record)
    rows: list[Example] = []
    for problem, found in sorted(by_problem.items()):
        if not found["right"] or not found["wrong"]:
            continue
        key = hashlib.sha256(problem.encode()).hexdigest()
        rng = random.Random(f"{seed}-{key}")
        right, wrong = solution_text(rng.choice(found["right"])), solution_text(rng.choice(found["wrong"]))
        if not right or not wrong or right == wrong:
            continue
        first = int(key, 16) % 2 == 0
        a, b, label = (right, wrong, "A") if first else (wrong, right, "B")
        rows.append({"id": f"prm800k-{key[:16]}", "suite": "prm800k", "family": f"prm800k-{key[:16]}",
                     "state": f"Question: {problem}\nResponse A: {a}\nResponse B: {b}", "question": dict(QUESTION),  # type: ignore[arg-type]
                     "label": label, "target": label,
                     "source": {"dataset": "prm800k", "license": "MIT", "rows": f"{SOURCE[0]}@{SOURCE[1]}/{SOURCE[2]}"}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/prm800k"))
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    path = Path(hf_hub_download(SOURCE[0], SOURCE[2], repo_type="dataset", revision=SOURCE[1]))
    records = [json.loads(line) for line in path.read_text().split("\n") if line]
    rows = pairs(records, args.seed)
    validate(rows)
    print(json.dumps({"rows": len(rows), "sha256": write_rows(args.out / "train.jsonl", rows)}))


if __name__ == "__main__":
    main()

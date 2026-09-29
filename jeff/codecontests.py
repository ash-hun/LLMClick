"""JudgeBench-style coding pairs from CodeContests: an accepted and a rejected Python 3 submission to the same problem.

CodeContests (DeepMind, CC BY 4.0; deepmind/code_contests on Hugging Face) holds competitive-programming problems with
real human submissions that the contest's judge accepted ("solutions") or rejected ("incorrect_solutions"). For a
problem with at least one of each in Python 3, one accepted and one rejected program become responses A and B (which is
which follows a hash of the problem), asking which answers the question more correctly - JudgeBench's coding items ask
the same. The judge's verdict is the label, so no code is run here."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

from jeff.data import validate, write_rows
from jeff.documents import MAX_TOKENS, TOKENIZER
from jeff.sampled import QUESTION
from jeff.types import Example

SOURCE = ("deepmind/code_contests", "802411c3010cb00d1b05bad57ca77365a3c699d6")
PYTHON3 = 3  # CodeContests' language code for Python 3
LIMIT = 4000
MAX_PROGRAM_CHARS = 3000


def python3(submissions: dict) -> list[str]:
    return [code for language, code in zip(submissions["language"], submissions["solution"])
            if language == PYTHON3 and code.strip() and len(code) <= MAX_PROGRAM_CHARS]


def pair(problem: dict, seed: int) -> Example | None:
    right, wrong = python3(problem["solutions"]), python3(problem["incorrect_solutions"])
    if not right or not wrong:
        return None
    key = hashlib.sha256(problem["name"].encode()).hexdigest()
    rng = random.Random(f"{seed}-{key}")
    good, bad = rng.choice(right).strip(), rng.choice(wrong).strip()
    if good == bad:
        return None
    first = int(key, 16) % 2 == 0
    a, b, label = (good, bad, "A") if first else (bad, good, "B")
    question = f"Write a Python 3 program that solves this problem.\n\n{problem['description'].strip()}"
    return {"id": f"codecontests-{key[:16]}", "suite": "codecontests", "family": f"codecontests-{key[:16]}",
            "state": f"Question: {question}\nResponse A: ```python\n{a}\n```\nResponse B: ```python\n{b}\n```",
            "question": dict(QUESTION), "label": label, "target": label,  # type: ignore[arg-type]
            "source": {"dataset": "codecontests", "license": "CC-BY-4.0", "rows": f"{SOURCE[0]}@{SOURCE[1]}",
                       "problem": problem["name"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/codecontests"))
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    files = sorted(s.rfilename for s in HfApi().dataset_info(SOURCE[0], revision=SOURCE[1]).siblings
                   if s.rfilename.startswith("data/train-"))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    rows: list[Example] = []
    too_long = 0
    for name in files:
        table = pq.read_table(hf_hub_download(SOURCE[0], name, repo_type="dataset", revision=SOURCE[1]),
                              columns=["name", "description", "solutions", "incorrect_solutions"])
        for problem in table.to_pylist():
            made = pair(problem, args.seed)
            if made is None:
                continue
            if len(tokenizer.encode(made["state"])) > MAX_TOKENS:  # type: ignore[arg-type]
                too_long += 1
                continue
            rows.append(made)
    rows = sorted(random.Random(args.seed).sample(rows, min(LIMIT, len(rows))), key=lambda r: r["id"])
    validate(rows)
    print(json.dumps({"rows": len(rows), "left_out_too_long": too_long, "sha256": write_rows(args.out / "train.jsonl", rows)}))


if __name__ == "__main__":
    main()

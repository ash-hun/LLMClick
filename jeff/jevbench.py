"""Convert JevBench's public hard tier into evaluation rows (a sixth benchmark next to the panel; never training data).

JevBench (github.com/fstandhartinger/jevbench, MIT) is an independent benchmark for Jev-style decision models. Its
public hard items are already in the decision format: a state and a question. The evaluator scores choice and
yes/no questions only, as AutoJev's did, so the few score questions are left out and counted in the manifest."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from jeff.data import validate, write_rows
from jeff.types import Example

REPOSITORY = "https://github.com/fstandhartinger/jevbench"
COMMIT = "d06ee95988da1350eb8ae5511daa0e06bfffe911"
PATH = "datasets/public/hard.jsonl"
SUITE = "JevBench public hard"


def convert(raw: dict) -> Example | None:
    """One JevBench item as an evaluation row; None for score questions."""
    question = raw["question"]
    if question["type"] == "score":
        return None
    if question["type"] == "noul":
        if raw["expected"] not in ("yes", "no"):
            raise ValueError(f"{raw['id']}: yes/no item expects {raw['expected']!r}")
        label: str | bool = raw["expected"] == "yes"
    elif question["type"] == "choice":
        if raw["expected"] not in question["criteria"]:
            raise ValueError(f"{raw['id']}: expected {raw['expected']!r} is not an option")
        label = raw["expected"]
    else:
        raise ValueError(f"{raw['id']}: unknown question type {question['type']!r}")
    return {"id": f"jevbench-{raw['id']}", "suite": SUITE, "family": f"jevbench-{raw['group']}", "state": raw["state"],
            "question": question, "label": label, "target": label,
            "source": {"dataset": "jevbench", "repository": REPOSITORY, "commit": COMMIT, "path": PATH, "license": "MIT",
                       "item_family": raw["family"], "original_id": raw["id"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path("data/jevbench/repo"),
                        help="A clone of the JevBench repository; the pinned commit is read from it")
    parser.add_argument("--out", type=Path, default=Path("data/jevbench-hard.jsonl"))
    args = parser.parse_args()
    text = subprocess.run(["git", "-C", str(args.checkout), "show", f"{COMMIT}:{PATH}"], check=True,
                          capture_output=True, text=True).stdout
    raws = [json.loads(line) for line in text.split("\n") if line]
    rows = [row for row in map(convert, raws) if row is not None]
    validate(rows)
    digest = write_rows(args.out, rows)
    manifest = {"repository": REPOSITORY, "commit": COMMIT, "path": PATH, "items": len(raws), "rows": len(rows),
                "left_out_score_questions": len(raws) - len(rows), "sha256": digest,
                "by_family": dict(Counter(row["source"]["item_family"] for row in rows)),
                "by_type": dict(Counter(row["question"]["type"] for row in rows))}
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""LegalBench tasks with a permissive licence, as decision rows (rule application, contract and statute reading).

LegalBench (github.com/HazyResearch/legalbench; rows on Hugging Face as nguha/legalbench) has 162 small legal
reasoning tasks written or collected by lawyers, each under its own licence. Only tasks under CC BY 4.0, CC BY-SA 4.0
or MIT are used; the non-commercial ones are left out, as are the tasks built from CUAD, MAUD and ContractNLI (used
directly elsewhere) and tasks whose README has no one-line summary. Each task's one-line summary becomes the question and its task description the background; its distinct
answers (two to eight of them) become the options. 15% of every task (by a hash of the row) is held back as a check set."""

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from jeff.data import validate, write_rows
from jeff.documents import MAX_TOKENS, TOKENIZER, fits
from jeff.types import Example

GITHUB = ("https://github.com/HazyResearch/legalbench", "b46bf4ffae90524b2b72aaa30e7745fe9db64481", "data/docs-raw/legalbench")
HUGGING_FACE = ("nguha/legalbench", "daec8237410aa23e3faf4bc41ad8b3a7e1696826")
PERMISSIVE = re.compile(r"CC By 4\.0|CC BY 4\.0|CC by 4\.0|CC BY-SA 4\.0|\bMIT\b")
NON_COMMERCIAL = re.compile(r"NonCommercial|BY-NC", re.IGNORECASE)
SKIPPED_PREFIXES = ("cuad_", "maud_", "contract_nli_")  # built from datasets used directly elsewhere
MAX_ANSWERS = 8
# Columns that are not the case itself: row numbers, reference ids, the category a row belongs to ("slice"), and the
# intermediate answers some tasks record (the diversity tasks' "parties_are_diverse" and "aic_is_met" are the two
# findings the question asks the model to work out), so keeping them would give the answer away.
EXCLUDED_COLUMNS = {"", "index", "idx", "slice", "parties_are_diverse", "aic_is_met", "case id", "year", "Docket No."}
PER_TASK = 400


def licence(readme: str) -> str | None:
    """The task's licence line when it is permissive; None otherwise."""
    line = re.search(r"(?im)^\s*\**licen[sc]e\**\s*:?\**\s*(.+)$", readme)
    text = line.group(1) if line else ""
    return text.strip() if PERMISSIVE.search(text) and not NON_COMMERCIAL.search(text) else None


def summary_and_background(readme: str) -> tuple[str, str]:
    """The one-line summary (the ### heading) and the first paragraph of the task description."""
    summary = re.search(r"(?m)^###\s+(.+)$", readme)
    description = re.search(r"(?s)## Task description\s*\n(.+?)(?=\n#|\Z)", readme)
    if summary is None:
        raise ValueError("LegalBench README has no ### summary line")
    return summary.group(1).strip(), description.group(1).strip() if description else ""


def slug(answer: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", answer.lower()).strip("_")[:40] or "empty"


def task_rows(name: str, readme: str, table: str) -> dict[str, list[Example]] | None:
    """A task's rows split into train and check; None when its licence or answers do not fit."""
    licensed = licence(readme)
    if licensed is None or name.startswith(SKIPPED_PREFIXES):
        return None
    csv.field_size_limit(10 ** 9)
    records = list(csv.DictReader(io.StringIO(table), delimiter="\t"))
    choice_columns = sorted(c for c in records[0] if re.fullmatch(r"choice_\d+", c))
    if choice_columns:  # multiple choice (SCALR): the answer is the index of one choice column
        keys = {str(i): f"choice_{i}" for i in range(len(choice_columns))}
        answers = sorted(keys)
    else:
        answers = sorted({r["answer"].strip() for r in records})
        keys = {answer: slug(answer) for answer in answers}
    if (not 2 <= len(answers) <= MAX_ANSWERS or len(set(keys.values())) != len(answers)
            or any(a.startswith("[") for a in answers)):  # lists of answers are multi-label, not one decision
        return None
    summary, background = summary_and_background(readme)
    inputs = [c for c in records[0] if c != "answer" and c not in EXCLUDED_COLUMNS and c not in choice_columns]
    instructions = f"{background}\n\nTask: {summary}" if background else summary
    split: dict[str, list[Example]] = {"train": [], "check": []}
    for number, record in enumerate(records[:PER_TASK * 2]):
        state: object = record[inputs[0]] if len(inputs) == 1 else {c: record[c] for c in inputs}
        digest = hashlib.sha256(f"{name}-{number}-{json.dumps(state)}".encode()).hexdigest()
        which = "check" if int(digest[:8], 16) / 16 ** 8 < 0.15 else "train"
        if which == "train" and len(split["train"]) >= PER_TASK:
            continue
        label = keys[record["answer"].strip()]
        split[which].append({"id": f"legalbench-{name}-{number}", "suite": "legalbench", "family": f"legalbench-{name}",
                             "state": state,  # type: ignore[typeddict-item]
                             "question": {"type": "choice", "instructions": instructions,
                                          "criteria": ({keys[a]: record[f"choice_{a}"] for a in answers} if choice_columns
                                                       else {keys[a]: a for a in answers})},
                             "label": label, "target": label,
                             "source": {"dataset": "legalbench", "task": name, "license": licensed,
                                        "rows": f"{HUGGING_FACE[0]}@{HUGGING_FACE[1]}", "readme": f"{GITHUB[0]}@{GITHUB[1]}"}})
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/legalbench"))
    args = parser.parse_args()
    checkout = Path(GITHUB[2])
    commit = __import__("subprocess").run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True,
                                          text=True).stdout.strip()
    if commit != GITHUB[1]:
        raise ValueError(f"{checkout} is at {commit}, not the pinned {GITHUB[1]}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    rows: dict[str, list[Example]] = {"train": [], "check": []}
    used, skipped = [], Counter()
    for task in sorted(p for p in (checkout / "tasks").iterdir() if (p / "README.md").exists()):
        readme = (task / "README.md").read_text(errors="replace")
        if licence(readme) is None:
            skipped["licence"] += 1
            continue
        if task.name.startswith(SKIPPED_PREFIXES):
            skipped["built from CUAD, MAUD or ContractNLI"] += 1
            continue
        if not re.search(r"(?m)^###\s+\S", readme):
            skipped["no summary line in README"] += 1
            continue
        table = Path(hf_hub_download(HUGGING_FACE[0], f"data/{task.name}/test.tsv", repo_type="dataset",
                                     revision=HUGGING_FACE[1])).read_text(encoding="utf-8")
        made = task_rows(task.name, readme, table)
        if made is None:
            skipped["answers not 2-8 single options"] += 1
            continue
        used.append(task.name)
        for which in rows:
            rows[which].extend(made[which])
    manifest: dict[str, object] = {"github": f"{GITHUB[0]}@{GITHUB[1]}", "hugging_face": f"{HUGGING_FACE[0]}@{HUGGING_FACE[1]}",
                                   "tasks_used": used, "tasks_skipped": dict(skipped), "max_tokens": MAX_TOKENS}
    for which, found in rows.items():
        kept, too_long = fits(found, tokenizer)
        validate(kept)
        manifest[which] = {"rows": len(kept), "left_out_too_long": too_long, "sha256": write_rows(args.out / f"{which}.jsonl", kept)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tasks_used"} | {"tasks_used": len(used)}, indent=2))


if __name__ == "__main__":
    main()

"""Questions with known answers for grounded pairwise judging: the teacher writes one correct and one wrong response,
and code checks each response's final answer against the known answers, so the label never depends on a judgement."""

import argparse
import json
import random
import re
from pathlib import Path
from typing import TypedDict

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from jeff.families import Slot

GSM8K = ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/train-00000-of-00001.parquet")
MULTIPLE_CHOICE = ("commonsense_qa", "arc_challenge", "arc_easy", "openbookqa", "qasc")
# Long responses need a wrong answer that can be argued convincingly at length: maths (a slip in a calculation) and
# harder science questions. Commonsense items with absurd distractors made the writer think aloud inside the text.
LONG_SOURCES = ("gsm8k-", "extra-arc_challenge-")
FINAL = re.compile(r"Final answer:\s*(.+)", re.IGNORECASE)


class Item(TypedDict):
    id: str
    question: str
    correct_answer: str
    wrong_answer: str


def number(text: str) -> float | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", text.replace("$", ""))
    return float(match.group(0).replace(",", "")) if match else None


def same_answer(given: str, expected: str) -> bool:
    expected_number = number(expected) if re.fullmatch(r"-?[\d,]+(?:\.\d+)?", expected.strip()) else None
    if expected_number is not None:
        return number(given) == expected_number
    normal = lambda text: " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())
    return normal(given) == normal(expected) or normal(given).endswith(" " + normal(expected))


def wrong_number(answer: int, rng: random.Random) -> int:
    """A plausible wrong result: a nearby value, a doubled or halved value, or an off-by-a-step value."""
    candidates = {answer + d for d in (-10, -5, -2, -1, 1, 2, 5, 10)} | {answer * 2}
    if answer % 2 == 0:
        candidates.add(answer // 2)
    candidates = {c for c in candidates if c != answer and (c >= 0 or answer < 0)}  # stay non-negative unless the answer is
    return rng.choice(sorted(candidates))


def gsm8k_items() -> list[Item]:
    rows = pq.read_table(hf_hub_download(GSM8K[0], GSM8K[2], repo_type="dataset", revision=GSM8K[1])).to_pylist()
    items: list[Item] = []
    for index, row in enumerate(rows):
        answer = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        if not re.fullmatch(r"-?\d+", answer):
            raise ValueError(f"GSM8K row {index}: final answer {answer!r} is not a whole number")
        rng = random.Random(f"gsm8k-{index}")
        items.append({"id": f"gsm8k-{index}", "question": row["question"], "correct_answer": answer,
                      "wrong_answer": str(wrong_number(int(answer), rng))})
    return items


def multiple_choice_items(extra: Path) -> list[Item]:
    items: list[Item] = []
    for line in extra.read_text().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if row["suite"] not in MULTIPLE_CHOICE:
            continue
        options = row["question"]["criteria"]
        rng = random.Random(f"mc-{row['id']}")
        wrong_key = rng.choice([key for key in options if key != row["label"]])
        stem = row["state"] if row["suite"] in ("commonsense_qa", "arc_challenge", "arc_easy", "openbookqa") else row["question"]["instructions"]
        listed = "\n".join(f"{key}) {text}" for key, text in options.items())
        items.append({"id": row["id"], "question": f"{stem}\n{listed}",
                      "correct_answer": f"{row['label']}) {options[row['label']]}", "wrong_answer": f"{wrong_key}) {options[wrong_key]}"})
    return items


def dress(slots: list[Slot], items: list[Item], seed: int) -> list[Slot]:
    """Give each grounded pairwise slot (short or long) its own question with known right and wrong answers."""
    dressed: list[Slot] = []
    for slot in slots:
        if slot["family"] in ("grounded_pairwise", "long_pairwise"):
            pool = items if slot["family"] == "grounded_pairwise" else [i for i in items if i["id"].startswith(LONG_SOURCES)]
            if not pool:
                raise ValueError(f"No grounded items from {LONG_SOURCES} for long pairwise slots")
            item = random.Random(f"{seed}-grounded-{slot['id']}").choice(pool)
            slot = {**slot, "question": item["question"], "correct_answer": item["correct_answer"], "wrong_answer": item["wrong_answer"]}
        dressed.append(slot)
    return dressed


def final_answers(text: str) -> dict[str, str | None]:
    """The last 'Final answer:' line inside each of Response A and Response B."""
    parts = re.split(r"\bResponse ([AB]):", text)
    answers: dict[str, str | None] = {"A": None, "B": None}
    for letter, body in zip(parts[1::2], parts[2::2]):
        found = FINAL.findall(body)
        answers[letter] = found[-1].strip() if found else None
    return answers


def problems(slot: Slot, text: str) -> list[str]:
    """What is wrong with a grounded-pairwise text, in words the writer can act on. Empty means it passes."""
    right, other = slot["label"], "B" if slot["label"] == "A" else "A"
    answers = final_answers(text)
    issues = [f"Response {letter} has no 'Final answer:' line" for letter in "AB" if answers[letter] is None]
    if not issues:
        if not same_answer(str(answers[right]), slot["correct_answer"]):
            issues.append(f"Response {right} must end with 'Final answer: {slot['correct_answer']}'")
        if not same_answer(str(answers[other]), slot["wrong_answer"]):
            issues.append(f"Response {other} must end with 'Final answer: {slot['wrong_answer']}'")
    return issues


def load(path: Path) -> list[Item]:
    # split("\n"), not splitlines(): JSON strings may contain Unicode line separators such as U+2028.
    items: list[Item] = [json.loads(line) for line in path.read_text().split("\n") if line]
    if not items:
        raise ValueError(f"{path} has no grounded items")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", type=Path, default=Path("data/extra/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/grounded.jsonl"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to replace {args.out}")
    items = gsm8k_items() + multiple_choice_items(args.extra)
    args.out.write_text("".join(json.dumps(item) + "\n" for item in items))
    print(json.dumps({"items": len(items), "gsm8k": sum(i["id"].startswith("gsm8k") for i in items)}))


if __name__ == "__main__":
    main()

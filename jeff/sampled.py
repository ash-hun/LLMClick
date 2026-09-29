"""JudgeBench-style pairs from honest attempts by the local teacher.

JudgeBench pairs two real model answers to a hard question with a known answer: one right, one wrong. We do the same
with the local teacher: it answers competition maths problems (MATH, MIT licence) several times, sincerely, with a
different sampling seed each time. Each attempt's final answer is compared with the known answer as a number, so the
label never depends on a model's judgement. A problem where one attempt is right and another wrong gives one pair.
Only problems whose known answer is a plain number (an integer, decimal or fraction) are used, so that grading is an
exact comparison. The wrong answers are natural mistakes, not written to order."""

import argparse
import asyncio
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from jeff.data import validate, write_rows
from jeff.teacher import TEACHER_MODEL, teacher_url, MalformedOutput, Teacher
from jeff.types import Example

SOURCE = ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3")
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus")
LEVELS = ("Level 3", "Level 4", "Level 5")
ATTEMPTS = 4
TEMPERATURE = 1.0
FAMILY = "sampled_pairwise"
SYSTEM = """You solve maths problems. Explain your solution step by step, in full sentences, the way a careful tutor
would: restate what is asked, work through each step, and check your result. End with a final line of the form
"Final answer: <answer>", giving the answer as a plain number or fraction in simplest form (for example 12, -3.5 or 7/4).
Return JSON only, with your whole solution, including the final line, in "solution"."""
# The same question JudgeBench asks (and the pairwise families use).
QUESTION = {"type": "choice", "instructions": "Which response answers the question more correctly?",
            "criteria": {"A": "Response A is more correct.", "B": "Response B is more correct."}}
SCHEMA = {"type": "object", "properties": {"solution": {"type": "string"}}, "required": ["solution"], "additionalProperties": False}


def boxed(solution: str) -> str | None:
    """The contents of the last \\boxed{...} in a MATH reference solution (braces may nest)."""
    start = solution.rfind("\\boxed{")
    if start < 0:
        return None
    depth, index, out = 1, start + len("\\boxed{"), []
    while index < len(solution) and depth:
        char = solution[index]
        depth += (char == "{") - (char == "}")
        if depth:
            out.append(char)
        index += 1
    return "".join(out) if depth == 0 else None


def as_number(text: str) -> Fraction | None:
    """A plain number, decimal or fraction (also LaTeX \\frac{a}{b}) as an exact value; None for anything else."""
    cleaned = re.sub(r"\\[dt]?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", text.replace("$", "").replace(" ", "").replace(",", ""))
    cleaned = cleaned.rstrip(".")
    match = re.fullmatch(r"(-?)(?:(\d+(?:\.\d+)?)|(\d+)/(\d+))", cleaned)
    if not match:
        return None
    sign = -1 if match.group(1) else 1
    if match.group(2) is not None:
        return sign * Fraction(match.group(2))
    return None if int(match.group(4)) == 0 else sign * Fraction(int(match.group(3)), int(match.group(4)))


def final_answer(solution: str) -> Fraction | None:
    found = re.findall(r"Final answer:\s*(.+)", solution)
    return as_number(found[-1].strip()) if found else None


def problems() -> list[dict]:
    """Level 3-5 MATH training problems whose known answer is a plain number."""
    result = []
    for subject in SUBJECTS:
        path = hf_hub_download(SOURCE[0], f"{subject}/train-00000-of-00001.parquet", repo_type="dataset", revision=SOURCE[1])
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            answer = boxed(row["solution"])
            value = as_number(answer) if answer is not None else None
            if row["level"] in LEVELS and value is not None:
                result.append({"id": f"math-{subject}-{index}", "problem": row["problem"], "answer": value,
                               "level": row["level"], "subject": subject})
    return result


def pair_row(problem: dict, right: str, wrong: str) -> Example:
    """The right and the wrong attempt as responses A and B; which is A follows a hash of the problem id."""
    first = int(hashlib.sha256(problem["id"].encode()).hexdigest(), 16) % 2 == 0
    a, b, label = (right, wrong, "A") if first else (wrong, right, "B")
    return {"id": f"syn-sampled-{problem['id']}", "suite": "synthetic", "family": f"syn-{FAMILY}-{problem['subject']}",
            "state": f"Question: {problem['problem']}\nResponse A: {a}\nResponse B: {b}", "question": dict(QUESTION),
            "label": label, "target": label,
            "source": {"dataset": "sampled-spark", "teacher": TEACHER_MODEL, "family": FAMILY, "problems": f"{SOURCE[0]}@{SOURCE[1]}",
                       "problem_id": problem["id"], "level": problem["level"], "subject": problem["subject"]}}


async def attempt(teacher: Teacher, problem: dict, seed: int) -> str | None:
    try:
        reply = await teacher.complete(system=SYSTEM, user=problem["problem"], schema=SCHEMA, temperature=TEMPERATURE,
                                       max_tokens=4000, seed=seed)
    except MalformedOutput:
        return None
    solution = reply.get("solution")
    return solution.strip() if isinstance(solution, str) and solution.strip() else None


async def process(teacher: Teacher, problem: dict) -> dict:
    """All attempts at one problem; a pair when at least one is right and one is wrong."""
    solutions = await asyncio.gather(*(attempt(teacher, problem, seed) for seed in range(ATTEMPTS)))
    graded = [(s, final_answer(s)) for s in solutions if s is not None]
    right = [s for s, value in graded if value == problem["answer"]]
    wrong = [s for s, value in graded if value is not None and value != problem["answer"]]
    outcome = {"problem": problem["id"], "right": len(right), "wrong": len(wrong), "attempts": len(solutions),
               "time": datetime.now(timezone.utc).isoformat(), "row": None}
    if right and wrong:
        outcome["row"] = pair_row(problem, right[0], wrong[0])
    return outcome


async def run(out: Path, count: int, seed: int, concurrency: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "outcomes.jsonl"
    done = {json.loads(line)["problem"] for line in path.read_text().split("\n") if line} if path.exists() else set()
    chosen = random.Random(seed).sample(problems(), count)
    pending = [p for p in chosen if p["id"] not in done]
    lock = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        teacher = Teacher(client, url=teacher_url(), model=TEACHER_MODEL, cache=out / "cache", log=out / "teacher.log",
                          max_in_flight=concurrency)
        gate = asyncio.Semaphore(concurrency)

        async def one(problem: dict) -> None:
            async with gate:
                outcome = await process(teacher, problem)
            async with lock:
                with path.open("a") as stream:
                    stream.write(json.dumps(outcome, ensure_ascii=False, default=str) + "\n")

        await asyncio.gather(*(one(p) for p in pending))


def finalize(out: Path) -> dict[str, object]:
    outcomes = [json.loads(line) for line in (out / "outcomes.jsonl").read_text().split("\n") if line]
    rows = sorted((o["row"] for o in outcomes if o["row"] is not None), key=lambda r: r["id"])
    validate(rows)
    report = {"problems": len(outcomes), "pairs": len(rows), "sha256": write_rows(out / "synthetic.jsonl", rows),
              "all_right": sum(o["wrong"] == 0 and o["right"] > 0 for o in outcomes),
              "all_wrong": sum(o["right"] == 0 for o in outcomes)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    go = commands.add_parser("run")
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--problems", type=int, required=True)
    go.add_argument("--seed", type=int, default=20260927)
    go.add_argument("--concurrency", type=int, required=True,
                    help="Requests in flight to the shared teacher; with another generator running, the two must add up to 8")
    done = commands.add_parser("finalize")
    done.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "finalize":
        print(json.dumps(finalize(args.out), indent=2))
        return
    asyncio.run(run(args.out, args.problems, args.seed, args.concurrency))


if __name__ == "__main__":
    main()

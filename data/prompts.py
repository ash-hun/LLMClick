"""Stage `prompts`: instructions from seeds, through a generator; answers come later, in `respond`."""

import re
from pathlib import Path
from typing import Any, ClassVar

from core.registry import Registry
from core.stage import Outputs, Stage
from data.config import SyntheticConfig
from data.rows import Row, cached, keep, key, read_rows, trail, write_rows
from data.teachers import Teacher, teacher_from

GENERATORS = Registry("generator")  # (teacher, seed row, index, settings, call) -> prompt rows
PROMPTS = "prompts.jsonl"
NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$")


def lines_of(text: str) -> list[str]:
    """Numbered or bulleted lines of a teacher's answer, without the markers."""
    found = []
    for line in text.split("\n"):
        match = NUMBERED.match(line)
        if match and match.group(1):
            found.append(match.group(1))
    return found


@GENERATORS.register("passthrough")
def passthrough(teacher: Teacher, seed: Row, index: int, params: dict[str, Any], ask: Any) -> list[Row]:
    """The seed text is already the instruction; `answer_field` names a column with the reference answer."""
    row = seed.get("row") or {}
    answer = row.get(params.get("answer_field", "answer")) if params.get("answer_field") else row.get("answer")
    return [{"instruction": seed["text"], **({"answer": str(answer)} if answer is not None else {})}]


@GENERATORS.register("topic_questions")
def topic_questions(teacher: Teacher, seed: Row, index: int, params: dict[str, Any], ask: Any) -> list[Row]:
    """Nemotron's open Q&A prompt: `n` questions about a topic, then each made more specific."""
    count = int(params.get("n", 3))
    language = params.get("language", "English")
    first = ask([{"role": "user", "content": f"Can you generate {count} questions or requests related to {seed['text']}? "
                                             f"Write them in {language}, one per line, numbered. Only the questions."}])
    questions = lines_of(first)[:count] or [first.strip()]
    rows = []
    for question in questions:
        refined = ask([{"role": "user", "content": f"Rewrite this question to be more detailed and specific, in {language}. "
                                                   f"Reply with the rewritten question only.\n\n{question}"}]).strip()
        rows.append({"instruction": refined or question})
    return rows


@GENERATORS.register("persona_task")
def persona_task(teacher: Teacher, seed: Row, index: int, params: dict[str, Any], ask: Any) -> list[Row]:
    """Tulu 3's persona-driven prompt: one realistic request this persona would make, about one skill."""
    skills = params.get("skills") or ["a question or task from their daily work"]
    skill = skills[(index + sum(ord(c) for c in seed["id"])) % len(skills)]
    language = params.get("language", "English")
    text = ask([{"role": "user", "content": f"Persona: {seed['text']}\n\nWrite one realistic request this person would make to an "
                                            f"AI assistant, about {skill}. Write it in {language}, in the first person, as they "
                                            f"would type it. Reply with the request only."}]).strip()
    return [{"instruction": text}] if text else []


@GENERATORS.register("document_qa")
def document_qa(teacher: Teacher, seed: Row, index: int, params: dict[str, Any], ask: Any) -> list[Row]:
    """A question answerable from the passage, with its answer, for grounded rows; the passage becomes `context`."""
    language = params.get("language", "English")
    text = ask([{"role": "user", "content": f"Passage:\n{seed['text']}\n\nWrite one question that the passage answers and its "
                                            f"answer, in {language}. Format exactly:\nQ: <question>\nA: <answer>"}])
    question = re.search(r"Q:\s*(.+)", text)
    answer = re.search(r"A:\s*(.+)", text, re.DOTALL)
    if not question or not answer:
        return []
    return [{"instruction": question.group(1).strip(), "answer": answer.group(1).strip(), "context": seed["text"]}]


class Calls:
    """The teacher calls one generator job makes, with their cost; one per job, so the cache entry holds the bill."""

    def __init__(self, teacher: Teacher, max_tokens: int, temperature: float, seed: int) -> None:
        self.teacher, self.max_tokens, self.temperature, self.seed = teacher, max_tokens, temperature, seed
        self.cost = {"input": 0, "output": 0, "usd": 0.0, "calls": 0}

    def __call__(self, messages: list[dict[str, str]]) -> str:
        completion = self.teacher.complete(messages, max_tokens=self.max_tokens, temperature=self.temperature, seed=self.seed)
        self.cost["input"] += completion.input_tokens
        self.cost["output"] += completion.output_tokens
        self.cost["usd"] += completion.cost_usd
        self.cost["calls"] += 1
        return completion.text


class PromptsStage(Stage[SyntheticConfig]):
    """Per seed and index, a generator call cached under the prompt's natural key, so a rerun after a crash makes
    only the prompts that are missing and the same ones again."""
    name: ClassVar[str] = "prompts"
    requires: ClassVar[tuple[str, ...]] = ("seeds",)
    sections: ClassVar[tuple[str, ...]] = ("prompts", "seed")

    def identity(self) -> Any:
        return [super().identity(), teacher_from(self.config.teacher).identity()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config, settings = self.config, self.config.prompts
        teacher = teacher_from(config.teacher)
        generator = GENERATORS.get(settings.generator)
        seeds = read_rows(Path(inputs["seeds"]["rows"]))
        rows: list[Row] = []
        calls = 0
        jobs = [(seed, index) for seed in seeds for index in range(settings.per_seed)]
        for done, (seed, index) in enumerate(jobs):
            self.progress.update(done, len(jobs), settings.generator)
            identifier = key("prompt", seed["id"], settings.generator, index, settings.params)
            entry = cached(workdir / "cache", identifier)
            if entry is None:
                ask = Calls(teacher, settings.max_tokens, settings.temperature, config.seed + done)
                made = generator(teacher, seed, index, settings.params, ask)
                entry = keep(workdir / "cache", identifier, {"made": made, "cost": ask.cost})
                calls += 1
            for position, made_row in enumerate(entry["made"]):
                instruction = str(made_row["instruction"]).strip()
                if settings.constraints:
                    instruction += " " + settings.constraints[sum(ord(c) for c in identifier) % len(settings.constraints)]
                rows.append({**made_row, "id": key("prompt", identifier, position), "seed_id": seed["id"], "instruction": instruction})
        self.progress.update(len(jobs), len(jobs))
        rows.sort(key=lambda row: row["id"])
        write_rows(workdir / PROMPTS, rows)
        summary = {"count": len(rows), "seeds": len(seeds), "generated_now": calls, **spend(workdir)}
        return {"rows": str(workdir / PROMPTS), **summary, "trail": trail(inputs, self.name, summary)}


def spend(workdir: Path) -> dict[str, Any]:
    """Tokens and cost of every cached call in a stage directory, built or reused: the ledger is the cache itself."""
    total = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}
    for path in sorted((workdir / "cache").glob("*.json")) if (workdir / "cache").is_dir() else []:
        entry = cached(workdir / "cache", path.stem) or {}
        cost = entry.get("cost") or {}
        total["input_tokens"] += int(cost.get("input", 0))
        total["output_tokens"] += int(cost.get("output", 0))
        total["cost_usd"] += float(cost.get("usd", 0.0))
        total["calls"] += int(cost.get("calls", 0))
    return total

"""Stage `respond`: the teacher answers every prompt N times; each answer is cached under its key, the budget is
checked before every call, and a rerun continues where the last one stopped."""

from pathlib import Path
from typing import Any, ClassVar

from core.stage import Outputs, Stage
from data.config import SyntheticConfig
from data.evolve import prompts_of
from data.prompts import spend
from data.rows import Row, cached, keep, key, read_rows, trail, write_rows
from data.teachers import teacher_from

CANDIDATES = "candidates.jsonl"


class BudgetExceeded(RuntimeError):
    pass


def messages_for(row: Row, system: str | None) -> list[dict[str, str]]:
    content = f"{row['context']}\n\n{row['instruction']}" if row.get("context") else str(row["instruction"])
    return [*([{"role": "system", "content": system}] if system else []), {"role": "user", "content": content}]


class RespondStage(Stage[SyntheticConfig]):
    name: ClassVar[str] = "respond"
    sections: ClassVar[tuple[str, ...]] = ("respond", "seed")

    def dependencies(self) -> tuple[str, ...]:
        return ("evolve",) if self.config.evolve is not None else ("prompts",)

    def identity(self) -> Any:
        return [super().identity(), teacher_from(self.config.teacher).identity()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config, settings = self.config, self.config.respond
        teacher = teacher_from(config.teacher)
        prompts = read_rows(prompts_of(inputs))
        limit = config.budget.max_usd
        spent = spend(workdir)["cost_usd"]
        rows: list[Row] = []
        jobs = [(prompt, sample) for prompt in prompts for sample in range(settings.answers_per_question)]
        for done, (prompt, sample) in enumerate(jobs):
            self.progress.update(done, len(jobs), f"answering, {spent:.2f} USD")
            identifier = key("candidate", prompt["id"], sample, teacher.identity(), settings.model_dump(mode="json"))
            entry = cached(workdir / "cache", identifier)
            if entry is None:
                if limit is not None and spent >= limit:
                    raise BudgetExceeded(f"budget.max_usd={limit} reached after {spent:.2f} USD; {len(jobs) - done} answers left. "
                                         "Raise the budget and run again: what is answered stays.")
                completion = teacher.complete(messages_for(prompt, settings.system), max_tokens=settings.max_tokens,
                                              temperature=settings.temperature, seed=config.seed + sample)
                cost = {"input": completion.input_tokens, "output": completion.output_tokens, "usd": completion.cost_usd, "calls": 1}
                entry = keep(workdir / "cache", identifier, {"text": completion.text, "cost": cost})
                spent += completion.cost_usd
            rows.append({"id": identifier, "prompt_id": prompt["id"], "seed_id": prompt.get("seed_id"), "sample": sample,
                         "instruction": prompt["instruction"], **({"context": prompt["context"]} if prompt.get("context") else {}),
                         **({"answer": prompt["answer"]} if prompt.get("answer") is not None else {}), "text": entry["text"]})
        self.progress.update(len(jobs), len(jobs))
        rows.sort(key=lambda row: (row["prompt_id"], row["sample"]))
        write_rows(workdir / CANDIDATES, rows)
        summary = {"count": len(rows), "prompts": len(prompts), **spend(workdir)}
        return {"rows": str(workdir / CANDIDATES), **summary, "trail": trail(inputs, self.name, summary)}

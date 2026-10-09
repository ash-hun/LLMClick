"""Stage `verify`: cheap checks first (schema, the sorry rule, a reward, majority agreement), the judge last; every
rejection keeps its reason."""

import re
from pathlib import Path
from typing import Any, ClassVar

from core.stage import Outputs, Stage
from data.config import SyntheticConfig
from data.prompts import spend
from data.respond import CANDIDATES
from data.rows import Row, cached, keep, key, normalized, read_rows, trail, write_rows
from data.teachers import Teacher, teacher_from
from modeling.llm.methods.grpo import REWARDS, last_number

VERIFIED = "verified.jsonl"
REJECTED = "rejected.jsonl"
SCORE = re.compile(r"\d+(?:\.\d+)?")
JUDGE = ("{rubric}\n\nRequest:\n{instruction}\n\nAnswer:\n{text}\n\nReply with a single integer score from 1 (worst) to 10 (best) "
         "and nothing else.")


def sorry_short(text: str, words: int) -> bool:
    """Evol-Instruct rule 2: an apology under `words` words means the prompt could not be answered."""
    return "sorry" in text.lower() and len(text.split()) < words


def extracted(text: str, how: str) -> str | None:
    if how == "last_number":
        number = last_number(text)
        return None if number is None else repr(number)
    return normalized(text)


def majority_keys(rows: list[Row], how: str) -> dict[str, str | None]:
    """Per prompt, the most common extracted answer among its candidates (ties broken by the smaller key)."""
    by_prompt: dict[str, dict[str, int]] = {}
    for row in rows:
        value = extracted(row["text"], how)
        if value is not None:
            counts = by_prompt.setdefault(row["prompt_id"], {})
            counts[value] = counts.get(value, 0) + 1
    return {prompt: max(sorted(counts), key=lambda v: counts[v]) if counts else None for prompt, counts in by_prompt.items()}


def assembled(row: Row, recipe: str) -> Row:
    """The candidate as the target recipe's row, for its `check`."""
    content = f"{row['context']}\n\n{row['instruction']}" if row.get("context") else row["instruction"]
    if recipe in {"llm_sft", "llm_decision_sft"}:
        return {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": row["text"]}]}
    if recipe == "llm_instruction":
        return {"instruction": row["instruction"], **({"input": row["context"]} if row.get("context") else {}), "output": row["text"]}
    if recipe == "llm_grpo":
        return {"prompt": content, "answer": row.get("answer", "")}
    raise ValueError(f"verify.schema_recipe {recipe!r} is not a recipe this stage can assemble")


def schema_error(row: Row, recipe: str) -> str | None:
    from core.registry import RECIPES, load_channels
    load_channels()
    config_class = RECIPES.get(recipe).config_class
    method = config_class.method_class(config_class.model_fields["method"].default_factory(), None)
    try:
        method.check(assembled(row, recipe))
    except (KeyError, TypeError, ValueError) as error:
        return str(error)
    return None


class VerifyStage(Stage[SyntheticConfig]):
    name: ClassVar[str] = "verify"
    requires: ClassVar[tuple[str, ...]] = ("respond",)
    sections: ClassVar[tuple[str, ...]] = ("verify", "seed")

    def identity(self) -> Any:
        judge = self.config.verify.judge
        return [super().identity(), teacher_from(judge.teacher or self.config.teacher).identity() if judge else None]

    def judged(self, workdir: Path, teacher: Teacher, row: Row) -> float:
        settings = self.config.verify.judge
        assert settings is not None
        identifier = key("judge", row["id"], teacher.identity(), settings.rubric)
        entry = cached(workdir / "cache", identifier)
        if entry is None:
            completion = teacher.complete([{"role": "user", "content": JUDGE.format(rubric=settings.rubric, instruction=row["instruction"], text=row["text"])}],
                                          max_tokens=settings.max_tokens, temperature=0.0, seed=self.config.seed)
            match = SCORE.search(completion.text)
            entry = keep(workdir / "cache", identifier, {"score": float(match.group()) if match else 0.0, "raw": completion.text,
                                                         "cost": {"input": completion.input_tokens, "output": completion.output_tokens,
                                                                  "usd": completion.cost_usd, "calls": 1}})
        return float(entry["score"])

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config, settings = self.config, self.config.verify
        rows = read_rows(Path(inputs["respond"]["rows"]))
        judge = teacher_from(settings.judge.teacher or config.teacher) if settings.judge else None
        reward = REWARDS.get(settings.reward.name) if settings.reward else None
        majority = majority_keys(rows, settings.majority) if settings.majority else {}
        verified: list[Row] = []
        rejected: list[Row] = []
        reasons: dict[str, int] = {}

        def reject(row: Row, reason: str) -> None:
            reasons[reason] = reasons.get(reason, 0) + 1
            rejected.append({**row, "rejected": reason})

        for done, row in enumerate(rows):
            self.progress.update(done, len(rows), "verifying")
            if not row["text"].strip():
                reject(row, "empty")
                continue
            if sorry_short(row["text"], settings.sorry_words):
                reject(row, "sorry")
                continue
            if settings.schema_recipe and (error := schema_error(row, settings.schema_recipe)):
                reject(row, f"schema: {error}")
                continue
            if reward is not None:
                if row.get("answer") is None:
                    reject(row, "reward: no reference answer")
                    continue
                row["reward"] = float(reward(row["text"], {"answer": row["answer"]}, settings.reward.params if settings.reward else {}))
                if row["reward"] <= 0:
                    reject(row, "reward")
                    continue
            if settings.majority:
                agreed = majority.get(row["prompt_id"])
                if agreed is None or extracted(row["text"], settings.majority) != agreed:
                    reject(row, "majority")
                    continue
            if judge is not None:
                row["score"] = self.judged(workdir, judge, row)
                if row["score"] < settings.judge.min_score:  # type: ignore[union-attr]
                    reject(row, "judge")
                    continue
            verified.append(row)
        self.progress.update(len(rows), len(rows))
        write_rows(workdir / VERIFIED, verified)
        write_rows(workdir / REJECTED, rejected)
        summary = {"count": len(verified), "rejected": len(rejected), "reasons": reasons, **spend(workdir)}
        return {"rows": str(workdir / VERIFIED), "rejected_rows": str(workdir / REJECTED), **summary, "trail": trail(inputs, self.name, summary)}


__all__ = ["CANDIDATES", "REJECTED", "VERIFIED", "VerifyStage", "assembled", "majority_keys", "sorry_short"]

"""Stage `evolve`: Evol-Instruct's operators make a prompt harder; the elimination rules drop the failures."""

import random
from pathlib import Path
from typing import Any, ClassVar

from core.registry import Registry
from core.stage import Outputs, Stage
from data.config import SyntheticConfig
from data.prompts import PROMPTS, spend
from data.rows import Row, cached, keep, key, normalized, read_rows, trail, write_rows
from data.teachers import teacher_from

EVOLVERS = Registry("evolver")  # name -> the method sentence the rewriter prompt takes
EVOLVED = "prompts.evolved.jsonl"
# WizardLM's rewriter and creator prompts (nlpxucan/WizardLM, Evol_Instruct/depth.py and breadth.py), kept verbatim.
REWRITER = ("I want you act as a Prompt Rewriter.\n"
            "Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems "
            "(e.g., chatgpt and GPT4) a bit harder to handle.\n"
            "But the rewritten prompt must be reasonable and must be understood and responded by humans.\n"
            "Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. "
            "Also, please do not omit the input in #The Given Prompt#.\n"
            "You SHOULD complicate the given prompt using the following method:\n{method}\n"
            "You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only "
            "add 10 to 20 words into #The Given Prompt#.\n"
            "'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to "
            "appear in #Rewritten Prompt#\n#The Given Prompt#:\n{prompt}\n#Rewritten Prompt#:\n")
CREATOR = ("I want you act as a Prompt Creator.\n"
           "Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\n"
           "This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\n"
           "The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\n"
           "The #Created Prompt# must be reasonable and must be understood and responded by humans.\n"
           "'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in "
           "#Created Prompt#\n#Given Prompt#:\n{prompt}\n#Created Prompt#:\n")
SCAFFOLDING = ("given prompt", "rewritten prompt", "created prompt", "#rewritten prompt#", "#the given prompt#", "#created prompt#")
GAIN = ("Does the second prompt add information, constraints or difficulty that the first one lacks? Answer with one "
        "word: yes or no.\n\nFirst prompt:\n{before}\n\nSecond prompt:\n{after}")

for _name, _method in {
    "constraints": "Please add one more constraints/requirements into #The Given Prompt#'",
    "deepen": "If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.",
    "concretize": "Please replace general concepts with more specific concepts.",
    "reasoning": "If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.",
    # the paper's fifth in-depth operator; its sentence is not in the repository, so this one is ours
    "complicate": "Please add a concrete input (a short table, a code snippet or a few data points) that the prompt must work with.",
    # WizardCoder's code heuristics, as one operator each
    "code_requirement": "Replace a commonly used requirement in the programming task with a less common and more specific one.",
    "code_misdirection": "Provide a piece of erroneous code as a reference to increase misdirection.",
    "code_complexity": "Propose higher time or space complexity requirements, but please refrain from doing so frequently.",
    # WizardMath's downward evolution
    "easier": "Please revise the question to a lower difficulty level, or produce a new and easier question about another topic.",
}.items():
    EVOLVERS.register(_name)(_method)
EVOLVERS.register("breadth")(None)


def evolve_prompt(operator: str, prompt: str) -> str:
    method = EVOLVERS.get(operator)
    return CREATOR.format(prompt=prompt) if method is None else REWRITER.format(method=method, prompt=prompt)


def failed(before: str, after: str) -> str | None:
    """Evol-Instruct's rules 3 and 4, and an empty or unchanged rewrite; rule 2 (the answer says sorry) is applied
    to answers in `verify`, rule 1 (no information gain) needs a judge and is optional."""
    text = normalized(after)
    if not text or not any(c.isalnum() for c in text):
        return "empty"
    if any(word in text for word in SCAFFOLDING):
        return "scaffolding"
    if text == normalized(before):
        return "unchanged"
    return None


class EvolveStage(Stage[SyntheticConfig]):
    """Every prompt, every round: one operator drawn from the weights with the prompt's key as the seed, one
    teacher call cached under the evolved key. All rounds are kept next to the originals, as the paper does."""
    name: ClassVar[str] = "evolve"
    requires: ClassVar[tuple[str, ...]] = ("prompts",)
    sections: ClassVar[tuple[str, ...]] = ("evolve", "seed")

    def identity(self) -> Any:
        return [super().identity(), teacher_from(self.config.teacher).identity()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config = self.config
        assert config.evolve is not None
        settings, teacher = config.evolve, teacher_from(config.teacher)
        names = [name for name, weight in settings.operators.items() if weight > 0]
        weights = [settings.operators[name] for name in names]
        originals = read_rows(Path(inputs["prompts"]["rows"]))
        rows: list[Row] = list(originals)
        parents, dropped = list(originals), {"empty": 0, "scaffolding": 0, "unchanged": 0, "no_gain": 0}
        total = len(originals) * settings.rounds
        done = 0
        for round_number in range(1, settings.rounds + 1):
            children: list[Row] = []
            for parent in parents:
                self.progress.update(done, total, f"round {round_number}")
                done += 1
                operator = random.Random(key("draw", parent["id"], round_number, config.seed)).choices(names, weights)[0]
                identifier = key("evolve", parent["id"], round_number, operator)
                entry = cached(workdir / "cache", identifier)
                if entry is None:
                    completion = teacher.complete([{"role": "user", "content": evolve_prompt(operator, parent["instruction"])}],
                                                  max_tokens=settings.max_tokens, temperature=settings.temperature, seed=config.seed + done)
                    cost = {"input": completion.input_tokens, "output": completion.output_tokens, "usd": completion.cost_usd, "calls": 1}
                    reason = failed(parent["instruction"], completion.text)
                    if reason is None and settings.judge_gain:
                        verdict = teacher.complete([{"role": "user", "content": GAIN.format(before=parent["instruction"], after=completion.text.strip())}],
                                                   max_tokens=8, temperature=0.0, seed=config.seed)
                        cost["input"] += verdict.input_tokens
                        cost["output"] += verdict.output_tokens
                        cost["usd"] += verdict.cost_usd
                        cost["calls"] += 1
                        reason = None if verdict.text.strip().lower().startswith("yes") else "no_gain"
                    entry = keep(workdir / "cache", identifier, {"text": completion.text.strip(), "failed": reason, "cost": cost})
                if entry["failed"]:
                    dropped[entry["failed"]] += 1
                    continue
                children.append({**parent, "id": identifier, "instruction": entry["text"], "parent_id": parent["id"],
                                 "round": round_number, "operator": operator})
            rows += children
            parents = children
        self.progress.update(total, total)
        rows.sort(key=lambda row: row["id"])
        write_rows(workdir / EVOLVED, rows)
        summary = {"count": len(rows), "originals": len(originals), "dropped": dropped, **spend(workdir)}
        return {"rows": str(workdir / EVOLVED), **summary, "trail": trail(inputs, self.name, summary)}


def prompts_of(inputs: dict[str, Outputs]) -> Path:
    """The prompts a later stage reads: evolved when the recipe evolved, the originals otherwise."""
    return Path(inputs["evolve"]["rows"]) if "evolve" in inputs else Path(inputs["prompts"]["rows"])


__all__ = ["EVOLVED", "EVOLVERS", "EvolveStage", "PROMPTS", "evolve_prompt", "failed", "prompts_of"]

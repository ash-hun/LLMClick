"""Config of the data recipes that start from seeds: the teacher, the seeds, and one section per stage."""

from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from core.config.schema import BaseConfig, Section
from modeling.config import Keyed, TrackerConfig, local_paths


class SeedSpec(Section):
    """Where seeds come from: inline values, or rows of a source with the text in `field`."""
    kind: str = Field(description="What the seeds are: topic, persona, document, instruction, ...; generators read it")
    values: list[str] | None = Field(default=None, description="Inline seed texts")
    source: Keyed | None = Field(default=None, description="A row source (local_jsonl, huggingface, ...)")
    field: str | None = Field(default=None, description="With `source`: the column that holds the seed text")
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_of(self) -> "SeedSpec":
        if (self.values is None) == (self.source is None):
            raise ValueError("a seed gives values or a source, not both")
        if self.source is not None and self.field is None:
            raise ValueError("a seed from a source needs `field`")
        return self


class PromptSettings(Section):
    generator: str = Field(description="Key of GENERATORS: passthrough, topic_questions, persona_task, document_qa")
    per_seed: int = Field(default=1, ge=1, description="Prompts to make from each seed")
    params: dict[str, Any] = Field(default_factory=dict, description="Generator parameters, e.g. skills for persona_task")
    constraints: list[str] = Field(default_factory=list, description="Verifiable constraints; one is appended to every prompt")
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=1.0, ge=0)


class EvolveSettings(Section):
    rounds: int = Field(default=1, ge=1)
    operators: dict[str, float] = Field(default_factory=lambda: {"constraints": 1.0, "deepen": 1.0, "concretize": 1.0, "reasoning": 1.0, "breadth": 1.0},
                                        description="EVOLVERS keys and their draw weights")
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    judge_gain: bool = Field(default=False, description="Ask the teacher whether the evolved prompt adds information (Evol-Instruct rule 1)")


class RespondSettings(Section):
    answers_per_question: int = Field(default=1, ge=1)
    system: str | None = None
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.7, ge=0)


class JudgeSettings(Section):
    teacher: Keyed | None = Field(default=None, description="Another model family than the generator; null: the generator")
    rubric: str = Field(default="Rate how helpful, correct and complete the answer is for the request.")
    min_score: float = Field(default=6, ge=1, le=10)
    max_tokens: int = Field(default=64, ge=1)


class VerifySettings(Section):
    schema_recipe: str | None = Field(default=None, description="A modeling recipe whose `check` every assembled row must pass")
    reward: Keyed | None = Field(default=None, description="A REWARDS function scoring the answer against the prompt's `answer`")
    majority: Literal["last_number", "exact"] | None = Field(default=None, description="Keep answers that agree with the majority of a prompt's answers")
    judge: JudgeSettings | None = None
    sorry_words: int = Field(default=80, ge=1, description="An answer with 'sorry' and fewer words than this is a failed prompt (Evol-Instruct rule 2)")


class DedupSettings(Section):
    near: float | None = Field(default=0.75, ge=0, le=1, description="Share of word n-grams shared with a kept row that makes a duplicate; null: exact only")
    ngram: int = Field(default=5, ge=1)


class LeakSettings(Section):
    against: list[Keyed] = Field(min_length=1, description="Row sources of the evaluation items the rows must not contain")
    threshold: float = Field(default=0.5, ge=0, le=1)
    ngram: int = Field(default=8, ge=1)


class DecisionQuestion(Section):
    """The question a `decision` row asks about each prompt (its `state`); the prompt's `answer` is the label."""
    name: str = Field(default="route", min_length=1, description="The question's key in the row's `questions`")
    instructions: str = Field(min_length=1, description="What the decision model is asked about the state")
    criteria: dict[str, str] = Field(min_length=2, description="Option -> what it means; the labels are these options")


class SelectSettings(Section):
    assemble: Literal["sft", "dpo", "grpo", "decision"] = "sft"
    decision: DecisionQuestion | None = Field(default=None, description="assemble: decision, or verify.schema_recipe: "
                                                                        "llm_decision_sft: the question every row asks")
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    leak: LeakSettings | None = None
    margin: float = Field(default=1.0, ge=0, description="dpo: the judge-score gap a chosen and rejected answer must have")
    max_rows: int | None = Field(default=None, ge=1)


class BudgetSettings(Section):
    max_usd: float | None = Field(default=None, ge=0, description="Stop before a teacher call would exceed this spend")


class SeedConfig(BaseConfig):
    """What every recipe that starts from seeds shares: the seeds, the prompts made from them, the model that makes them."""
    teacher: Keyed | None = Field(default=None, description="The model generators ask; null when the generator asks none (passthrough)")
    seeds: list[SeedSpec] = Field(min_length=1)
    prompts: PromptSettings

    @model_validator(mode="after")
    def _seeded(self) -> "SeedConfig":
        from data.prompts import GENERATORS
        from data.teachers import TEACHERS
        from modeling.tuning.sources import SOURCES
        TEACHERS.check(*([self.teacher.name] if self.teacher is not None else []))
        GENERATORS.check(self.prompts.generator)
        SOURCES.check(*(seed.source.name for seed in self.seeds if seed.source is not None))
        return self

    def paths(self) -> list[str]:
        return [*super().paths(), *local_paths([seed.source for seed in self.seeds if seed.source is not None])]


class SyntheticConfig(SeedConfig):
    identity_exclude: ClassVar[frozenset[str]] = BaseConfig.identity_exclude | {"tracker", "budget"}

    teacher: Keyed
    evolve: EvolveSettings | None = None
    respond: RespondSettings = Field(default_factory=RespondSettings)
    verify: VerifySettings = Field(default_factory=VerifySettings)
    select: SelectSettings = Field(default_factory=SelectSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)

    @model_validator(mode="after")
    def _registered(self) -> "SyntheticConfig":
        from data.evolve import EVOLVERS
        from data.teachers import TEACHERS
        from modeling.llm.methods.grpo import REWARDS
        if self.evolve is not None:
            unknown = [name for name in self.evolve.operators if name not in EVOLVERS]
            if unknown:
                raise ValueError(f"Unknown evolve operators {unknown}; registered: {EVOLVERS.names()}")
        REWARDS.check(*([self.verify.reward.name] if self.verify.reward is not None else []))
        if self.verify.judge is not None and self.verify.judge.teacher is not None and self.verify.judge.teacher.name not in TEACHERS:
            raise ValueError(f"Unknown judge teacher {self.verify.judge.teacher.name!r}; registered: {TEACHERS.names()}")
        if (self.select.assemble == "decision" or self.verify.schema_recipe == "llm_decision_sft") and self.select.decision is None:
            raise ValueError("decision rows need select.decision: the question, its instructions and the criteria of its options")
        return self

    def paths(self) -> list[str]:
        return [*super().paths(), *local_paths(self.select.leak.against if self.select.leak else [])]

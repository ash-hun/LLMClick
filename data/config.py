"""Config of `data_synthetic`: the teacher, the seeds, and one section per stage."""

from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from core.config.schema import BaseConfig, Section
from modeling.config import Keyed, TrackerConfig


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


class SelectSettings(Section):
    assemble: Literal["sft", "dpo", "grpo"] = "sft"
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    leak: LeakSettings | None = None
    margin: float = Field(default=1.0, ge=0, description="dpo: the judge-score gap a chosen and rejected answer must have")
    max_rows: int | None = Field(default=None, ge=1)


class BudgetSettings(Section):
    max_usd: float | None = Field(default=None, ge=0, description="Stop before a teacher call would exceed this spend")


class SyntheticConfig(BaseConfig):
    identity_exclude: ClassVar[frozenset[str]] = BaseConfig.identity_exclude | {"tracker", "budget"}

    teacher: Keyed
    seeds: list[SeedSpec] = Field(min_length=1)
    prompts: PromptSettings
    evolve: EvolveSettings | None = None
    respond: RespondSettings = Field(default_factory=RespondSettings)
    verify: VerifySettings = Field(default_factory=VerifySettings)
    select: SelectSettings = Field(default_factory=SelectSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)

    @model_validator(mode="after")
    def _registered(self) -> "SyntheticConfig":
        from data.evolve import EVOLVERS
        from data.prompts import GENERATORS
        from data.teachers import TEACHERS
        from modeling.llm.methods.grpo import REWARDS
        from modeling.tuning.sources import SOURCES
        if self.teacher.name not in TEACHERS:
            raise ValueError(f"Unknown teacher {self.teacher.name!r}; registered: {TEACHERS.names()}")
        if self.prompts.generator not in GENERATORS:
            raise ValueError(f"Unknown generator {self.prompts.generator!r}; registered: {GENERATORS.names()}")
        for seed in self.seeds:
            if seed.source is not None and seed.source.name not in SOURCES:
                raise ValueError(f"Unknown source {seed.source.name!r}; registered: {SOURCES.names()}")
        if self.evolve is not None:
            unknown = [name for name in self.evolve.operators if name not in EVOLVERS]
            if unknown:
                raise ValueError(f"Unknown evolve operators {unknown}; registered: {EVOLVERS.names()}")
        if self.verify.reward is not None and self.verify.reward.name not in REWARDS:
            raise ValueError(f"Unknown reward {self.verify.reward.name!r}; registered: {REWARDS.names()}")
        if self.verify.judge is not None and self.verify.judge.teacher is not None and self.verify.judge.teacher.name not in TEACHERS:
            raise ValueError(f"Unknown judge teacher {self.verify.judge.teacher.name!r}; registered: {TEACHERS.names()}")
        return self

    def paths(self) -> list[str]:
        own = [str(path) for seed in self.seeds if seed.source is not None and (path := seed.source.params.get("path")) is not None]
        leak = [str(path) for source in (self.select.leak.against if self.select.leak else []) if (path := source.params.get("path")) is not None]
        return [*super().paths(), *own, *leak]

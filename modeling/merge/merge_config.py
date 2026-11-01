"""Config of `llm_merge`: an LLM recipe whose `model` is the base, `merge` the models and rule, rows SFT conversations."""

from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from modeling.merge.merge_rule import RULES, TASK_RULES
from modeling.tuning.tuning_config import is_commit
from modeling.llm.methods.sft_method import SFT
from core.config.config_schema import Section
from modeling.llm.llm_config import LLMConfig


class MergedModel(Section):
    """One model to merge: a checkpoint directory (`path`) or a pinned Hub model (`name` and `revision`)."""
    path: str | None = Field(default=None, description="A checkpoint directory, e.g. output/<experiment>/validate/checkpoint; "
                                                        "checked when the merge runs, so it may come from an experiment not run yet")
    name: str | None = Field(default=None, description="A Hugging Face model ID")
    revision: str | None = Field(default=None, description="40-character commit of `name`")
    weight: float = Field(default=1.0, gt=0, description="Its share in linear, task_arithmetic, ties and dare")

    @model_validator(mode="after")
    def _one_source(self) -> "MergedModel":
        if (self.path is None) == (self.name is None):
            raise ValueError("a merged model is a checkpoint `path` or a Hub `name` with its `revision`, not both")
        if self.name is not None and not is_commit(self.revision):
            raise ValueError("a merged model's revision must be a 40-character commit hash (a branch name is not reproducible)")
        return self

    def source(self) -> tuple[str, str | None]:
        return (self.path, None) if self.path is not None else (str(self.name), self.revision)


class MergeSettings(Section):
    rule: str = Field(description="linear, slerp, task_arithmetic, ties, dare_linear or dare_ties")
    models: list[MergedModel] = Field(min_length=1)
    t: float = Field(default=0.5, ge=0, le=1, description="slerp: 0 is the first model, 1 the second")
    density: float = Field(default=0.5, gt=0, le=1, description="ties and dare: the share of each change that is kept")


class MergeConfig(LLMConfig):
    """`model` is the base the task rules take changes from; linear and slerp only read it for its architecture and
    for the held-out loss the validate stage compares against. Rows are SFT conversations."""
    method_class: ClassVar = SFT
    method: SFT.Config = Field(default_factory=SFT.Config)
    merge: MergeSettings

    @model_validator(mode="after")
    def _mergeable(self) -> "MergeConfig":
        RULES.check(self.merge.rule)
        if self.training.adapter is not None:
            raise ValueError("llm_merge trains nothing; remove training.adapter")
        if self.merge.rule == "slerp" and len(self.merge.models) != 2:
            raise ValueError("slerp interpolates between exactly two models")
        if self.merge.rule not in TASK_RULES and self.merge.rule != "slerp" and len(self.merge.models) < 2:
            raise ValueError(f"{self.merge.rule} needs at least two models")
        return self

    def paths(self) -> list[str]:
        return [*super().paths(), *(model.source()[0] for model in self.merge.models)]

    def local_models(self) -> list[str]:
        return [*super().local_models(), *(model.path for model in self.merge.models
                                           if model.path is not None and Path(model.path).is_dir())]

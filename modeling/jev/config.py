"""Pydantic schema of a jev config; one YAML is one custom model."""

import string
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from modeling.config import Keyed, ModelingConfig, ValidationConfig

PROMPT_LAYOUTS = ("state-first", "live-last")


class FoldsConfig(BaseModel):
    dev: int = Field(default=1500, ge=1, description="Rows carved from the builders' output for checkpoint selection")
    temperature: int = Field(default=1000, ge=1, description="Rows carved for fitting the calibration temperature")
    validation: int = Field(default=1000, ge=1, description="Rows training never sees, scored by the validate stage")


class TeacherConfig(Keyed):
    name: str = "openai_compatible"


class SyntheticConfig(BaseModel):
    enabled: bool = False
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    slots: int = Field(default=1000, ge=1)
    focus: Literal["v1", "v2", "v3", "v4", "v5", "bbh-code", "fallacies"] | None = None
    concurrency: int = Field(default=8, ge=1)


class MixConfig(BaseModel):
    size: int = Field(default=50000, ge=1, description="Public rows kept; synthetic rows are matched to this count")
    sweep_size: int = Field(default=10000, ge=1)
    panel_layout: bool = True
    escape: bool = True
    adversarial: bool = True


class DataConfig(BaseModel):
    builders: list[Keyed] = Field(min_length=1)
    folds: FoldsConfig = Field(default_factory=FoldsConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    mix: MixConfig = Field(default_factory=MixConfig)


class ModelConfig(BaseModel):
    backbone: str = "qwen3_5"
    name: str = Field(description="Hugging Face model ID")
    revision: str = Field(description="40-character commit of the base model")
    prompt_layout: Literal["state-first", "live-last"] | None = None

    @field_validator("revision")
    @classmethod
    def _immutable_revision(cls, value: str) -> str:
        if len(value) != 40 or any(c not in string.hexdigits for c in value):
            raise ValueError("model.revision must be a 40-character commit hash")
        return value


class TrainingConfig(BaseModel):
    """Field names match jeff.train arguments one to one."""
    epochs: int = Field(default=1, ge=1)
    lr: float = 5e-6
    weight_decay: float = 0.01
    batch_size: int = Field(default=32, ge=1)
    effective_batch_size: int = Field(default=256, ge=1)
    token_budget: int = Field(default=8192, ge=1)
    max_length: int = Field(default=8192, ge=1)
    eval_every: int = Field(default=40, ge=1)
    public_eval_every: int = Field(default=1_000_000, ge=1)
    resume_every: int = Field(default=50, ge=1)
    patience: int | None = Field(default=None, ge=1)
    stop_after: int | None = Field(default=None, ge=1, description="Pilot: stop after this many updates")
    cpu_threads: int = Field(default=16, ge=1)


class JevValidationConfig(ValidationConfig):
    """Bounds on accuracy, ece, brier and nll measured on the validation fold."""
    batch_size: int = Field(default=16, ge=1)


class EvaluationConfig(BaseModel):
    benchmarks: list[Keyed] = Field(min_length=1)
    batch_size: int = Field(default=16, ge=1)


class JevConfig(ModelingConfig):
    checkpoint: str | None = Field(default=None, description="Evaluate this checkpoint instead of training one")

    data: DataConfig | None = None
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    validation: JevValidationConfig = Field(default_factory=JevValidationConfig)
    evaluation: EvaluationConfig | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "JevConfig":
        from modeling.jev.registry import BACKBONES, BENCHMARKS, BUILDERS, TEACHERS, fill
        fill()
        if self.training is not None and (self.data is None or self.model is None):
            raise ValueError("training needs both data and model sections")
        if self.training is not None and self.checkpoint is not None:
            raise ValueError("Give either training (train a checkpoint) or checkpoint (evaluate one), not both")
        if self.evaluation is not None and self.training is None and self.checkpoint is None:
            raise ValueError("evaluation needs a checkpoint: add training or set checkpoint")
        if self.data is not None:
            for builder in self.data.builders:
                if builder.name not in BUILDERS:
                    raise ValueError(f"Unknown builder {builder.name!r}; registered: {BUILDERS.names()}")
            if self.data.synthetic.enabled and self.data.synthetic.teacher.name not in TEACHERS:
                raise ValueError(f"Unknown teacher {self.data.synthetic.teacher.name!r}; registered: {TEACHERS.names()}")
        if self.model is not None:
            if self.model.backbone not in BACKBONES:
                raise ValueError(f"Unknown backbone {self.model.backbone!r}; registered: {BACKBONES.names()}")
            if self.model.prompt_layout and not BACKBONES.get(self.model.backbone)["supports_prompt_layout"]:
                raise ValueError(f"backbone {self.model.backbone!r} does not support prompt_layout")
        if self.evaluation is not None:
            for benchmark in self.evaluation.benchmarks:
                if benchmark.name not in BENCHMARKS:
                    raise ValueError(f"Unknown benchmark {benchmark.name!r}; registered: {BENCHMARKS.names()}")
        return self

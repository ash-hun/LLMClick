"""Pydantic schema of the YAML config; one file is one experiment, and its hash names the experiment directory."""

import string
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROMPT_LAYOUTS = ("state-first", "live-last")
STAGES = ("data", "benchmarks", "synthetic", "mix", "train", "evaluate")


class Keyed(BaseModel):
    """A registry entry: `name` picks the implementation, every other key is passed to it as a parameter."""
    model_config = ConfigDict(extra="allow")
    name: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class FoldsConfig(BaseModel):
    dev: int = Field(default=1500, ge=1, description="Rows carved from the builders' output for checkpoint selection")
    temperature: int = Field(default=1000, ge=1, description="Rows carved for fitting the calibration temperature")


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


class TrackerConfig(BaseModel):
    enabled: bool = False
    project: str = "llmclick"
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    benchmarks: list[Keyed] = Field(min_length=1)
    batch_size: int = Field(default=16, ge=1)


class PipelineConfig(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    seed: int = 20260920
    output_dir: str = "./output"
    device: Literal["cuda", "mps", "cpu"] | None = None
    checkpoint: str | None = Field(default=None, description="Evaluate this checkpoint instead of training one")

    data: DataConfig | None = None
    model: ModelConfig | None = None
    training: TrainingConfig | None = None
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    evaluation: EvaluationConfig | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "PipelineConfig":
        from core.registry import BACKBONES, BENCHMARKS, BUILDERS, TEACHERS, catalogue
        catalogue()
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

    def section(self, name: str) -> dict[str, Any]:
        """A JSON-safe view of one section, used to fingerprint stages."""
        value = getattr(self, name)
        return value.model_dump(mode="json") if isinstance(value, BaseModel) else value

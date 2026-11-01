"""Config shared by every fine-tuning recipe: which backbone, which rows, and the optimisation settings."""

import string
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, model_validator

from modeling.modeling_config import Keyed, ModelingConfig, ValidationConfig, local_paths
from core.config.config_schema import Section
from core.recipe_registry import Registry

if TYPE_CHECKING:
    from modeling.tuning.tuning_backbone import Backbone
    from modeling.tuning.training_method import TrainingMethod


def is_commit(revision: str | None) -> bool:
    return revision is not None and len(revision) == 40 and all(c in string.hexdigits for c in revision)


class BackboneConfig(Section):
    architecture: str = Field(description="Key of the family's model catalog")
    name: str = Field(description="Hugging Face model ID, or a local directory")
    revision: str | None = Field(default=None, description="40-character commit; required unless `name` is a directory")
    init: str | None = Field(default=None, description="Start from this checkpoint directory of an earlier experiment")

    @model_validator(mode="after")
    def _pinned(self) -> "BackboneConfig":
        local = Path(self.name).is_dir()
        if not local and not is_commit(self.revision):
            raise ValueError("model.revision must be a 40-character commit hash (a branch name is not reproducible)")
        return self


class DataConfig(Section):
    sources: list[Keyed] = Field(min_length=1)
    validation: float = Field(default=0.1, gt=0, lt=1, description="Share of rows held out for the validate stage")


class AdapterConfig(Section):
    """Train small low-rank matrices next to the frozen weights instead of the weights themselves."""
    name: Literal["lora"] = "lora"
    r: int = Field(default=16, ge=1, description="Rank of the update matrices")
    alpha: int = Field(default=32, ge=1, description="Update scale; the effective factor is alpha / r")
    dropout: float = Field(default=0.0, ge=0, lt=1)
    targets: list[str] | None = Field(default=None, description="Layer names to adapt; default: the architecture's")
    added_keys: ClassVar[frozenset[str]] = frozenset({"quantize"})
    quantize: Literal["nf4", "int8"] | None = Field(default=None, description="QLoRA: load the frozen base weights in 4 or 8 bits (bitsandbytes, CUDA only)")


class TrainingConfig(Section):
    epochs: int = Field(default=1, ge=1)
    lr: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    batch_size: int = Field(default=8, ge=1)
    accumulation: int = Field(default=1, ge=1, description="Batches per optimizer step")
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    max_length: int = Field(default=1024, ge=8, description="Tokens per sequence")
    overflow: Literal["error", "skip"] = Field(default="error", description="Rows over max_length: stop before training, or leave them out")
    max_steps: int | None = Field(default=None, ge=1, description="Pilot: stop after this many optimizer steps")
    resume_every: int | None = Field(default=200, ge=1, description="Steps between resume snapshots; null disables them")
    eval_every: int | None = Field(default=None, ge=1, description="Steps between evaluations on the held-out rows while training; null: only validate at the end")
    eval_rows: int | None = Field(default=None, ge=1, description="How many held-out rows the in-training evaluation uses; null: all of them")
    adapter: AdapterConfig | None = Field(default=None, description="null trains every weight; set it to train LoRA")
    gradient_checkpointing: bool = Field(default=False, description="Recompute activations in the backward pass: less memory, about a third more compute")
    devices: int = Field(default=1, ge=1, le=64, description="Data-parallel processes, one per GPU; each takes a share of every batch")
    added_keys: ClassVar[frozenset[str]] = frozenset({"gradient_checkpointing", "devices"})

    @model_validator(mode="after")
    def _shares(self) -> "TrainingConfig":
        if self.devices > self.batch_size:
            raise ValueError(f"training.devices={self.devices} exceeds batch_size={self.batch_size}: some processes would get no rows")
        return self


class TuningValidationConfig(ValidationConfig):
    batch_size: int = Field(default=8, ge=1)


class TuningConfig(ModelingConfig):
    """A family sets `backbones`; a recipe sets `method_class` and types its own `method:` section."""
    backbones: ClassVar[Registry]
    method_class: ClassVar[type["TrainingMethod[Any]"]]

    model: BackboneConfig
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    validation: TuningValidationConfig = Field(default_factory=TuningValidationConfig)
    method: Any = None

    @model_validator(mode="after")
    def _registered(self) -> "TuningConfig":
        from modeling.tuning.row_sources import SOURCES
        if self.model.architecture not in self.backbones:
            raise ValueError(f"Unknown architecture {self.model.architecture!r}; registered: {self.backbones.names()}")
        SOURCES.check(*(source.name for source in self.data.sources))
        adapter = self.training.adapter
        if adapter is not None and adapter.quantize is not None and not self.backbones.get(self.model.architecture).quantizable:
            raise ValueError(f"training.adapter.quantize: architecture {self.model.architecture!r} cannot load quantized weights")
        return self

    def paths(self) -> list[str]:
        """`model.name` is a Hub ID or a directory; a Hub ID has no `..` or leading slash, so it passes as a path."""
        return [*super().paths(), self.model.name, *([self.model.init] if self.model.init else []),
                *local_paths(self.data.sources)]

    def local_files(self) -> list[str]:
        """Local files the backbone reads besides its weights (a tokens list, a chat template); their content joins
        the train fingerprint."""
        return []

    def local_models(self) -> list[str]:
        """Local model directories the method reads besides `model` (a distillation teacher); their content joins
        the train fingerprint like `model.init`'s."""
        return []

    def versions(self) -> dict[str, int]:
        """Code versions of the parts this config selects; they join the train fingerprint, so bumping one part's
        `version` rebuilds the experiments that use that part and no others."""
        return {"method": self.method_class.version, "backbone": self.backbones.get(self.model.architecture).version}

    def build_backbone(self) -> "Backbone":
        backbone: Backbone = self.backbones.get(self.model.architecture)(self.model)
        backbone.quantize = self.training.adapter.quantize if self.training.adapter is not None else None
        return backbone

    def build_method(self) -> "TrainingMethod[Any]":
        return self.method_class(self.method, self.training)

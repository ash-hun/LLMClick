"""Config shared by every fine-tuning recipe: which backbone, which rows, and the optimisation settings."""

import string
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, model_validator

from modeling.config import Keyed, ModelingConfig, ValidationConfig
from core.config.schema import Section
from core.registry import Registry

if TYPE_CHECKING:
    from modeling.tuning.backbone import Backbone
    from modeling.tuning.method import TrainingMethod


class BackboneConfig(Section):
    architecture: str = Field(description="Key of the family's model catalog")
    name: str = Field(description="Hugging Face model ID, or a local directory")
    revision: str | None = Field(default=None, description="40-character commit; required unless `name` is a directory")
    init: str | None = Field(default=None, description="Start from this checkpoint directory of an earlier experiment")

    @model_validator(mode="after")
    def _pinned(self) -> "BackboneConfig":
        local = Path(self.name).is_dir()
        commit = self.revision is not None and len(self.revision) == 40 and all(c in string.hexdigits for c in self.revision)
        if not local and not commit:
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
    adapter: AdapterConfig | None = Field(default=None, description="null trains every weight; set it to train LoRA")


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
        from modeling.tuning.sources import SOURCES
        if self.model.architecture not in self.backbones:
            raise ValueError(f"Unknown architecture {self.model.architecture!r}; registered: {self.backbones.names()}")
        for source in self.data.sources:
            if source.name not in SOURCES:
                raise ValueError(f"Unknown source {source.name!r}; registered: {SOURCES.names()}")
        return self

    def paths(self) -> list[str]:
        """`model.name` is a Hub ID or a directory; a Hub ID has no `..` or leading slash, so it passes as a path."""
        local = [str(path) for source in self.data.sources if (path := source.params.get("path")) is not None]
        return [*super().paths(), self.model.name, *([self.model.init] if self.model.init else []), *local]

    def versions(self) -> dict[str, int]:
        """Code versions of the parts this config selects; they join the train fingerprint, so bumping one part's
        `version` rebuilds the experiments that use that part and no others."""
        return {"method": self.method_class.version, "backbone": self.backbones.get(self.model.architecture).version}

    def build_backbone(self) -> "Backbone":
        backbone: Backbone = self.backbones.get(self.model.architecture)(self.model)
        return backbone

    def build_method(self) -> "TrainingMethod[Any]":
        return self.method_class(self.method, self.training)

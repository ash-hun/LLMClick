"""Config of the evaluation recipes: which trained model, which rows, how to measure."""

import string
from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from core.config.schema import BaseConfig, Section
from modeling.config import Keyed, TrackerConfig

BUILT = ("validate", "train", "base")  # `model.checkpoint` values that name a checkpoint of the experiment


class SourceModel(Section):
    """The model to score: one a modeling experiment built (`experiment` + `checkpoint`; its config says how it loads
    and how it is measured), or a Hub model or directory on its own (`name` + `revision`), which only the recipes
    that load the model themselves can score: benchmarks, and embeddings when `architecture` names the catalog entry."""
    experiment: str | None = Field(default=None, description="Experiment directory (output/<name>-<hash>) with its config.yaml")
    checkpoint: str = Field(default="validate", description="validate (passed validation), train (as trained), base "
                                                            "(the weights the experiment started from), or a checkpoint directory")
    name: str | None = Field(default=None, description="Instead of an experiment: a Hugging Face model ID or a local directory")
    revision: str | None = Field(default=None, description="40-character commit of `name`; required unless it is a directory")
    architecture: str | None = Field(default=None, description="With `name`: the model catalog entry (e.g. bi_encoder) for recipes that load the model themselves")

    @model_validator(mode="after")
    def _one_of(self) -> "SourceModel":
        if (self.experiment is None) == (self.name is None):
            raise ValueError("give exactly one of model.experiment or model.name")
        if self.name is not None and not Path(self.name).is_dir():
            commit = self.revision is not None and len(self.revision) == 40 and all(c in string.hexdigits for c in self.revision)
            if not commit:
                raise ValueError("model.revision must be a 40-character commit hash when model.name is a Hub model")
        return self

    @property
    def hub(self) -> bool:
        return self.name is not None

    @property
    def reference(self) -> str:
        """How messages name the model: the Hub ID or directory, or the experiment directory."""
        return self.name if self.name is not None else str(self.experiment)

    def paths(self) -> list[str]:
        if self.name is not None:
            return [self.name]
        return [str(self.experiment), *([] if self.checkpoint in BUILT else [self.checkpoint])]


class EvaluationData(Section):
    sources: list[Keyed] = Field(min_length=1, description="Rows in the format of the experiment's recipe")

    def paths(self) -> list[str]:
        return [str(path) for source in self.sources if (path := source.params.get("path")) is not None]


class EvaluationSettings(Section):
    batch_size: int = Field(default=8, ge=1)


class MeasureSettings(BaseConfig):
    """What every evaluation recipe shares: how the rows are measured, and whether the report goes to the tracker."""
    identity_exclude: ClassVar[frozenset[str]] = BaseConfig.identity_exclude | {"tracker"}

    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)


class EvaluationConfig(MeasureSettings):
    """`evaluation_custom`: one model, any rows, the metrics of the recipe that trained it."""
    model: SourceModel
    data: EvaluationData

    def paths(self) -> list[str]:
        return [*super().paths(), *self.model.paths(), *self.data.paths()]

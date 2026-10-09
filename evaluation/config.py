"""Config of the evaluation recipes: which trained model, which rows, how to measure."""

from typing import ClassVar

from pydantic import Field

from core.config.schema import BaseConfig, Section
from modeling.config import Keyed, TrackerConfig

BUILT = ("validate", "train", "base")  # `model.checkpoint` values that name a checkpoint of the experiment


class SourceModel(Section):
    """The model to score is one a modeling experiment built: its config says how it loads and how it is measured."""
    experiment: str = Field(description="Experiment directory (output/<name>-<hash>) with its config.yaml")
    checkpoint: str = Field(default="validate", description="validate (passed validation), train (as trained), base "
                                                            "(the weights the experiment started from), or a checkpoint directory")

    def paths(self) -> list[str]:
        return [self.experiment, *([] if self.checkpoint in BUILT else [self.checkpoint])]


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

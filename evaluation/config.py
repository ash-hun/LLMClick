"""Config of the evaluation recipes: which trained model, which rows, how to measure."""

from pydantic import Field

from core.config.schema import BaseConfig, Section
from modeling.config import Keyed

BUILT = ("validate", "train", "base")  # `model.checkpoint` values that name a checkpoint of the experiment


class SourceModel(Section):
    """The model to score is one a modeling experiment built: its config says how it loads and how it is measured."""
    experiment: str = Field(description="Experiment directory (output/<name>-<hash>) with its config.yaml")
    checkpoint: str = Field(default="validate", description="validate (passed validation), train (as trained), base "
                                                            "(the weights the experiment started from), or a checkpoint directory")


class EvaluationData(Section):
    sources: list[Keyed] = Field(min_length=1, description="Rows in the format of the experiment's recipe")


class EvaluationSettings(Section):
    batch_size: int = Field(default=8, ge=1)


class EvaluationConfig(BaseConfig):
    model: SourceModel
    data: EvaluationData
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    def paths(self) -> list[str]:
        local = [str(path) for source in self.data.sources if (path := source.params.get("path")) is not None]
        checkpoint = [] if self.model.checkpoint in BUILT else [self.model.checkpoint]
        return [*super().paths(), self.model.experiment, *checkpoint, *local]

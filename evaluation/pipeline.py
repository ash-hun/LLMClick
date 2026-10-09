"""The evaluation recipes: each measures a trained model and writes a report in one shape."""

from typing import ClassVar

from core.pipeline import Pipeline, recipe
from evaluation.config import BUILT, EvaluationConfig
from evaluation.stages import ReportStage, RowsStage, ScoreStage
from modeling.tuning.sources import SOURCES


class EvaluationPipeline(Pipeline[EvaluationConfig]):
    """Base of every evaluation recipe: the last stage writes `report.json`."""

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "source": SOURCES.names(), "checkpoint": [*BUILT, "<directory>"]}


@recipe
class CustomEvaluation(EvaluationPipeline):
    """Score an experiment's checkpoint on any rows with the metrics of the recipe that trained it."""
    kind: ClassVar[str] = "evaluation_custom"
    config_class = EvaluationConfig
    stage_classes = (RowsStage, ScoreStage, ReportStage)

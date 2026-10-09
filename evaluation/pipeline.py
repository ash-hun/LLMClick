"""The evaluation recipes: each measures a trained model and writes a report in one shape."""

from typing import Any, ClassVar

from core.pipeline import Pipeline, recipe
from core.stage import ConfigT, Stage
from evaluation.benchmark import PRESETS, BenchmarkConfig, BenchmarkReport, BenchmarkStage
from evaluation.config import BUILT, EvaluationConfig
from evaluation.stages import ReportStage, RowsStage, ScoreStage
from modeling.tuning.sources import SOURCES


class EvaluationPipeline(Pipeline[ConfigT]):
    """Base of every evaluation recipe: the last stage writes `report.json`."""

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "checkpoint": [*BUILT, "<directory>"]}


@recipe
class CustomEvaluation(EvaluationPipeline[EvaluationConfig]):
    """Score an experiment's checkpoint on any rows with the metrics of the recipe that trained it."""
    kind: ClassVar[str] = "evaluation_custom"
    config_class = EvaluationConfig
    stage_classes = (RowsStage, ScoreStage, ReportStage)

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "source": SOURCES.names()}


@recipe
class BenchmarkEvaluation(EvaluationPipeline[BenchmarkConfig]):
    """Score an experiment's checkpoint on public benchmarks with lm-evaluation-harness: one stage per benchmark,
    so a benchmark added later is the only one scored, and a comparison of several checkpoints shares every
    benchmark run that matches."""
    kind: ClassVar[str] = "evaluation_benchmark"
    config_class = BenchmarkConfig
    stage_classes = (BenchmarkStage, BenchmarkReport)  # the shape; `build_stages` makes one score stage per benchmark

    def build_stages(self) -> dict[str, Stage[Any]]:
        config: BenchmarkConfig = self.config
        scores = [BenchmarkStage.for_task(config, self.progress, task) for task in config.benchmarks.resolved()]
        report = BenchmarkReport(config, self.progress)
        report.scores = tuple(stage.name for stage in scores)
        return {**{stage.name: stage for stage in scores}, report.name: report}

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "preset": sorted(PRESETS)}

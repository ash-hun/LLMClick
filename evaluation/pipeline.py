"""The evaluation recipes: each measures a trained model and writes a report in one shape."""

from typing import Any, ClassVar

from core.pipeline import Pipeline, recipe
from core.stage import ConfigT, Stage
from evaluation.benchmark import PRESETS, BenchmarkConfig, BenchmarkReport, BenchmarkStage
from evaluation.compare import CompareConfig, CompareReport
from evaluation.config import BUILT, EvaluationConfig
from evaluation.decision import DecisionConfig, DecisionReport, DecisionScoreStage
from evaluation.embedding import EmbeddingConfig, EmbeddingReport, EmbeddingStage
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

    def build_stages(self) -> dict[str, Stage[Any]]:
        config: EvaluationConfig = self.config
        rows = RowsStage.for_(config, self.progress, config.data, config.model)
        score = ScoreStage.for_(config, self.progress, config.model)
        return {rows.name: rows, score.name: score, ReportStage.name: ReportStage(config, self.progress)}

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
        scores = [BenchmarkStage.for_(config, self.progress, task, config.model) for task in config.benchmarks.resolved()]
        report = BenchmarkReport(config, self.progress)
        report.scores = tuple(stage.name for stage in scores)
        return {**{stage.name: stage for stage in scores}, report.name: report}

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "preset": sorted(PRESETS)}


@recipe
class CompareEvaluation(EvaluationPipeline[CompareConfig]):
    """Several checkpoints, the base model among them, on the same benchmarks and rows. Every score stage is the
    one `evaluation_benchmark` or `evaluation_custom` would run, so results already measured are reused."""
    kind: ClassVar[str] = "evaluation_compare"
    config_class = CompareConfig
    stage_classes = (RowsStage, ScoreStage, BenchmarkStage, CompareReport)

    def build_stages(self) -> dict[str, Stage[Any]]:
        config: CompareConfig = self.config
        stages: dict[str, Stage[Any]] = {}
        report = CompareReport(config, self.progress)
        report.benchmark_stages, report.row_stages = {}, {}
        if config.benchmarks is not None:
            for run in config.runs:
                report.benchmark_stages[run.label] = {}
                for task in config.benchmarks.resolved():
                    stage = BenchmarkStage.for_(config, self.progress, task, run.model, f"score:{run.label}:{task.name}")
                    stages[stage.name] = stage
                    report.benchmark_stages[run.label][task.name] = stage.name
        if config.rows is not None:
            rows = RowsStage.for_(config, self.progress, config.rows, config.runs[0].model)
            stages[rows.name] = rows
            for run in config.runs:
                score = ScoreStage.for_(config, self.progress, run.model, f"rows:{run.label}", rows.name)
                stages[score.name] = score
                report.row_stages[run.label] = score.name
        stages[report.name] = report
        return stages

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "preset": sorted(PRESETS), "source": SOURCES.names()}


@recipe
class DecisionEvaluation(EvaluationPipeline[DecisionConfig]):
    """A decision model on Jev rows or the JevBench tiers (`jevbench` source): accuracy, NLL, ECE, per tier, without
    and with reasoning, with reasoning length and latency per question."""
    kind: ClassVar[str] = "evaluation_decision"
    config_class = DecisionConfig
    stage_classes = (RowsStage, DecisionScoreStage, DecisionReport)

    def build_stages(self) -> dict[str, Stage[Any]]:
        config: DecisionConfig = self.config
        rows = RowsStage.for_(config, self.progress, config.data, config.model)
        return {rows.name: rows, DecisionScoreStage.name: DecisionScoreStage(config, self.progress),
                DecisionReport.name: DecisionReport(config, self.progress)}

    @classmethod
    def catalogue(cls) -> dict[str, list[str]]:
        return {**super().catalogue(), "source": SOURCES.names(), "think": ["off", "on", "both"]}


@recipe
class EmbeddingEvaluation(EvaluationPipeline[EmbeddingConfig]):
    """An embedding model on MTEB tasks through the `mteb` package, one cached stage per task."""
    kind: ClassVar[str] = "evaluation_embedding"
    config_class = EmbeddingConfig
    stage_classes = (EmbeddingStage, EmbeddingReport)

    def build_stages(self) -> dict[str, Stage[Any]]:
        config: EmbeddingConfig = self.config
        scores = [EmbeddingStage.for_(config, self.progress, task) for task in config.tasks]
        report = EmbeddingReport(config, self.progress)
        report.scores = tuple(stage.name for stage in scores)
        return {**{stage.name: stage for stage in scores}, report.name: report}

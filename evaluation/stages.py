"""The stages of `evaluation_custom`: rows -> score -> report. The rows and score stages are built per model and
row set (`for_`), so a comparison can hold several of them; their scope makes the work shared across recipes."""

from pathlib import Path
from typing import Any, ClassVar

from core.progress import Progress
from core.stage import Outputs, Stage
from core.utils.device import resolve_device
from core.utils.files import write_json
from evaluation.config import EvaluationConfig, EvaluationData, MeasureSettings, SourceModel
from evaluation.contamination import measure, training_rows_of
from evaluation.source import SourceExperiment
from modeling.config import keyed_identity
from modeling.tuning.method import Row, fitting
from modeling.tuning.sources import SOURCES
from modeling.tuning.stages import read_rows, write_rows

SCORES = "scores.json"
REPORT = "report.json"


def named(cls: type[Stage[Any]], name: str) -> type[Any]:
    """A subclass with another stage name, so one stage class serves several instances of a pipeline."""
    return type(f"{cls.__name__}_{name}", (cls,), {"name": name})


class RowsStage(Stage[MeasureSettings]):
    """Read every source and check each row against the recipe of the experiment under evaluation."""
    name: ClassVar[str] = "rows"
    scope: ClassVar[str] = "rows"
    data: EvaluationData
    model: SourceModel

    @classmethod
    def for_(cls, config: MeasureSettings, progress: Progress, data: EvaluationData, model: SourceModel,
             name: str = "rows") -> "RowsStage":
        stage: RowsStage = named(cls, name)(config, progress)
        stage.data, stage.model = data, model
        return stage

    def identity(self) -> Any:
        source = SourceExperiment(self.model)
        return [keyed_identity(self.data.sources), source.config().recipe if source.exists() else None]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        sources = self.data.sources
        _, method, _ = SourceExperiment(self.model).build()
        rows: list[Row] = []
        for index, source in enumerate(sources):
            self.progress.update(index, len(sources), source.name)
            rows += SOURCES.get(source.name)(source.params)
        self.progress.update(len(sources), len(sources), "checking rows")
        for index, row in enumerate(rows):
            try:
                method.check(row)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Row {index} does not fit the experiment's recipe: {error}") from None
        if not rows:
            raise ValueError("no rows to score")
        write_rows(workdir / "rows.jsonl", rows)
        return {"rows": str(workdir / "rows.jsonl"), "count": len(rows)}


class ScoreStage(Stage[MeasureSettings]):
    """Load the checkpoint and measure it on the rows with the recipe's own `evaluate`."""
    name: ClassVar[str] = "score"
    scope: ClassVar[str] = "measure"
    sections: ClassVar[tuple[str, ...]] = ("evaluation", "device")
    model: SourceModel
    rows_stage: str = "rows"

    @classmethod
    def for_(cls, config: MeasureSettings, progress: Progress, model: SourceModel, name: str = "score",
             rows_stage: str = "rows") -> "ScoreStage":
        stage: ScoreStage = named(cls, name)(config, progress)
        stage.model, stage.rows_stage = model, rows_stage
        return stage

    def dependencies(self) -> tuple[str, ...]:
        return (self.rows_stage,)

    def identity(self) -> Any:
        return [super().identity(), SourceExperiment(self.model).identity()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        source = SourceExperiment(self.model)
        backbone, method, trained = source.build()
        checkpoint = source.checkpoint()
        self.progress.update(0, None, "loading checkpoint" if checkpoint else "loading base model")
        backbone.load(resolve_device(self.config.device), checkpoint)
        try:
            rows = read_rows(Path(inputs[self.rows_stage]["rows"]))
            kept = fitting(backbone, method, rows, trained.training, self.progress, "evaluation")
            metrics = method.evaluate(backbone, kept, self.config.evaluation.batch_size, self.progress)
        finally:
            backbone.release()
        scores = {"metrics": metrics, "rows": {"scored": len(kept), "skipped": len(rows) - len(kept)},
                  "checkpoint": str(checkpoint) if checkpoint else "base", "recipe": trained.recipe,
                  "settings": {"batch_size": self.config.evaluation.batch_size, "max_length": trained.training.max_length,
                               "device": backbone.device}}
        write_json(workdir / SCORES, scores)
        return scores


def contamination(model: SourceModel, items: list[Any]) -> dict[str, Any] | None:
    """Overlap of the items with what the experiment trained on; None for `base`, which those rows never trained."""
    if model.checkpoint == "base" or model.experiment is None:
        return None
    return measure(training_rows_of(Path(model.experiment)), items)


class ReportStage(Stage[EvaluationConfig]):
    """The report every evaluation recipe writes in the same shape, so comparisons can read any of them."""
    name: ClassVar[str] = "report"
    requires: ClassVar[tuple[str, ...]] = ("rows", "score")

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        scores = inputs["score"]
        self.progress.update(0, None, "measuring overlap with the training rows")
        report = {"model": {**SourceExperiment(self.config.model).described(), "checkpoint": scores["checkpoint"], "recipe": scores["recipe"]},
                  "settings": scores["settings"],
                  "rows": {"sources": [source.model_dump(mode="json") for source in self.config.data.sources], **scores["rows"]},
                  "scores": scores["metrics"],
                  "contamination": contamination(self.config.model, read_rows(Path(inputs["rows"]["rows"])))}
        write_json(workdir / REPORT, report)
        return {"report": str(workdir / REPORT), "scores": scores["metrics"]}

"""The stages of `evaluation_custom`: rows -> score -> report."""

from pathlib import Path
from typing import Any, ClassVar

from core.stage import Outputs, Stage
from core.utils.device import resolve_device
from core.utils.files import write_json
from evaluation.config import EvaluationConfig
from evaluation.source import SourceExperiment
from modeling.config import keyed_identity
from modeling.tuning.method import Row, fitting
from modeling.tuning.sources import SOURCES
from modeling.tuning.stages import read_rows, write_rows

SCORES = "scores.json"
REPORT = "report.json"


class RowsStage(Stage[EvaluationConfig]):
    """Read every source and check each row against the recipe of the experiment under evaluation."""
    name: ClassVar[str] = "rows"

    def identity(self) -> Any:
        source = SourceExperiment(self.config.model)
        return [keyed_identity(self.config.data.sources), source.config().recipe if source.exists() else None]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        sources = self.config.data.sources
        _, method, _ = SourceExperiment(self.config.model).build()
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


class ScoreStage(Stage[EvaluationConfig]):
    """Load the checkpoint and measure it on the rows with the recipe's own `evaluate`."""
    name: ClassVar[str] = "score"
    requires: ClassVar[tuple[str, ...]] = ("rows",)
    sections: ClassVar[tuple[str, ...]] = ("evaluation", "device")

    def identity(self) -> Any:
        return [super().identity(), SourceExperiment(self.config.model).identity()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        source = SourceExperiment(self.config.model)
        backbone, method, trained = source.build()
        checkpoint = source.checkpoint()
        self.progress.update(0, None, "loading checkpoint" if checkpoint else "loading base model")
        backbone.load(resolve_device(self.config.device), checkpoint)
        try:
            rows = read_rows(Path(inputs["rows"]["rows"]))
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


class ReportStage(Stage[EvaluationConfig]):
    """The report every evaluation recipe writes in the same shape, so comparisons can read any of them."""
    name: ClassVar[str] = "report"
    requires: ClassVar[tuple[str, ...]] = ("score",)

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        scores = inputs["score"]
        report = {"model": {"experiment": self.config.model.experiment, "checkpoint": scores["checkpoint"],
                            "recipe": scores["recipe"]},
                  "settings": scores["settings"],
                  "rows": {"sources": [source.model_dump(mode="json") for source in self.config.data.sources], **scores["rows"]},
                  "scores": scores["metrics"], "contamination": None}
        write_json(workdir / REPORT, report)
        return {"report": str(workdir / REPORT), "scores": scores["metrics"]}

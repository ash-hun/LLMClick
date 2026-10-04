"""The three stages every fine-tuning recipe runs: data -> train -> validate."""

import json
import hashlib
from pathlib import Path
from typing import Any, ClassVar

from modeling.tuning.loop import CHECKPOINT, SUMMARY, fit, history
from modeling.stages import TrainStage, ValidateStage
from modeling.tuning.sources import SOURCES
from modeling.tuning.config import TuningConfig
from modeling.tuning.method import Row, fitting
from modeling.config import keyed_identity
from core.utils.device import resolve_device
from core.utils.files import directory_signature, read_json
from modeling.tracker import History
from core.stage import Outputs, Stage

MINIMUM_ROWS = 2


def read_rows(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().split("\n") if line]


def write_rows(path: Path, rows: list[Row]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


class DataStage(Stage[TuningConfig]):
    """Read every source, check the rows against the method, and hold out the validation share."""
    name: ClassVar[str] = "data"

    def identity(self) -> Any:
        data = self.config.data
        return [self.config.seed, self.config.recipe, keyed_identity(data.sources), data.validation,
                self.config.build_method().check_identity()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        sources, method = self.config.data.sources, self.config.build_method()
        rows: list[Row] = []
        for index, source in enumerate(sources):
            self.progress.update(index, len(sources), source.name)
            rows += SOURCES.get(source.name)(source.params)
        self.progress.update(len(sources), len(sources), "checking rows")
        for index, row in enumerate(rows):
            try:
                method.check(row)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Row {index} does not fit recipe {self.config.recipe!r}: {error}") from None
        if len(rows) < MINIMUM_ROWS:
            raise ValueError(f"{len(rows)} rows cannot be split into training and validation")
        # A row's side depends only on the seed and its position, so adding sources never reshuffles earlier rows.
        ranked = sorted(range(len(rows)), key=lambda i: hashlib.sha256(f"{self.config.seed}:{i}".encode()).hexdigest())
        held_out = set(ranked[: min(len(rows) - 1, max(1, round(len(rows) * self.config.data.validation)))])
        split = {"train": [row for i, row in enumerate(rows) if i not in held_out],
                 "validation": [row for i, row in enumerate(rows) if i in held_out]}
        for name, fold in split.items():
            write_rows(workdir / f"{name}.jsonl", fold)
        return {**{name: str(workdir / f"{name}.jsonl") for name in split},
                "rows": {name: len(fold) for name, fold in split.items()}}


class TuneStage(TrainStage[TuningConfig]):
    """Load the backbone, let the method prepare, run the loop; a rerun resumes or returns the finished checkpoint."""
    requires: ClassVar[tuple[str, ...]] = ("data",)
    sections: ClassVar[tuple[str, ...]] = ("seed", "recipe", "model", "training", "method", "device")

    def identity(self) -> Any:
        """Local weights count by content, not by path: `model.init` usually goes through an experiment's link,
        which points somewhere else once that experiment is rebuilt."""
        model = self.config.model
        return [super().identity(), directory_signature(Path(model.init)) if model.init else None,
                directory_signature(Path(model.name)), self.config.versions()]

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        checkpoint, summary = workdir / CHECKPOINT, workdir / SUMMARY
        if not (summary.exists() and checkpoint.is_dir()):
            backbone, method, training = self.config.build_backbone(), self.config.build_method(), self.config.training
            self.progress.update(0, None, "loading model")
            backbone.load(resolve_device(self.config.device))
            try:
                # Both folds are checked now: a held-out row that does not fit would otherwise fail after training.
                fitting(backbone, method, read_rows(Path(inputs["data"]["validation"])), training, self.progress, "validation")
                rows = fitting(backbone, method, read_rows(Path(inputs["data"]["train"])), training, self.progress, "training")
                rows = method.prepare(backbone, rows, workdir, self.progress)
                backbone.adapt(training.adapter)
                fit(backbone, method, rows, training, workdir, self.config.seed, self.progress)
            finally:
                backbone.release()
        return {"checkpoint": str(checkpoint), "run": str(workdir), **read_json(summary)}

    def history(self, outputs: Outputs) -> History:
        return [(int(event["step"]), {f"train/{key}": float(value) for key, value in event.items() if key != "step"})
                for event in history(Path(outputs["run"]))]


class MeasureStage(ValidateStage[TuningConfig]):
    """The method's own metrics on the held-out rows, computed with the trained checkpoint."""
    requires: ClassVar[tuple[str, ...]] = ("train", "data")
    sections: ClassVar[tuple[str, ...]] = ("validation", "device")

    def identity(self) -> Any:
        training = self.config.training  # which held-out rows are measured depends on the length rule
        return [super().identity(), training.max_length, training.overflow]

    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        backbone, method = self.config.build_backbone(), self.config.build_method()
        self.progress.update(0, None, "loading checkpoint")
        backbone.load(resolve_device(self.config.device), Path(inputs["train"]["checkpoint"]))
        try:
            rows = fitting(backbone, method, read_rows(Path(inputs["data"]["validation"])), self.config.training,
                           self.progress, "validation")
            return method.evaluate(backbone, rows, self.config.validation.batch_size, self.progress)
        finally:
            backbone.release()

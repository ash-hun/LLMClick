"""The three stages every fine-tuning recipe runs: data -> train -> validate."""

from pathlib import Path
from typing import Any, ClassVar

from modeling.tuning.loop import CHECKPOINT, SUMMARY, fit, history, tracked
from modeling.stages import TrainStage, ValidateStage, held_out_positions
from modeling.tuning.sources import SOURCES
from modeling.tuning.config import TuningConfig
from modeling.tuning.method import Row, fitting
from modeling.config import keyed_identity
from core.utils.device import resolve_device
from core.utils.files import directory_signature, read_json, read_jsonl, write_jsonl
from modeling.tracker import History
from core.stage import Outputs, Stage

MINIMUM_ROWS = 2


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
        held_out = held_out_positions(len(rows), self.config.seed, self.config.data.validation)
        split = {"train": [row for i, row in enumerate(rows) if i not in held_out],
                 "validation": [row for i, row in enumerate(rows) if i in held_out]}
        for name, fold in split.items():
            write_jsonl(workdir / f"{name}.jsonl", fold)
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
            with backbone.loaded(resolve_device(self.config.device)):
                # Both folds are checked now: a held-out row that does not fit would otherwise fail after training.
                held_out = fitting(backbone, method, read_jsonl(Path(inputs["data"]["validation"])), training, self.progress, "validation")
                rows = fitting(backbone, method, read_jsonl(Path(inputs["data"]["train"])), training, self.progress, "training")
                rows = method.prepare(backbone, rows, workdir, self.progress)
                backbone.adapt(training.adapter)
                sample, batch_size = held_out[: training.eval_rows], self.config.validation.batch_size

                def evaluation() -> dict[str, float]:  # the validate stage's measurement, on the weights as they are
                    return method.evaluate(backbone, sample, batch_size, self.progress)

                with self.tracking(workdir) as log:
                    fit(backbone, method, rows, training, workdir, self.config.seed, self.progress, log, evaluation)
        return {"checkpoint": str(checkpoint), "run": str(workdir), **read_json(summary)}

    def history(self, outputs: Outputs) -> History:
        return [(int(event["step"]), tracked(event)) for event in history(Path(outputs["run"]))]


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
        with backbone.loaded(resolve_device(self.config.device), Path(inputs["train"]["checkpoint"])):
            rows = fitting(backbone, method, read_jsonl(Path(inputs["data"]["validation"])), self.config.training,
                           self.progress, "validation")
            return method.evaluate(backbone, rows, self.config.validation.batch_size, self.progress)

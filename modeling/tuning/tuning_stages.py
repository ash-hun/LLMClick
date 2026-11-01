"""The three stages every fine-tuning recipe runs: data -> train -> validate."""

from pathlib import Path
from typing import Any, ClassVar
from collections.abc import Callable

from modeling.tuning.training_loop import CHECKPOINT, SUMMARY, fit, history, tracked
from modeling.tuning.data_parallel import World, spawn
from modeling.modeling_stages import TrainStage, ValidateStage, held_out_positions
from modeling.tuning.row_sources import SOURCES
from modeling.tuning.tuning_config import TuningConfig
from modeling.tuning.training_method import Row, fitting
from modeling.modeling_config import keyed_identity
from core.device_selector import Device, resolve_device
from core.file_store import directory_signature, read_json, read_jsonl, sha256_file, write_jsonl
from modeling.metric_tracker import History
from core.progress_reporter import Progress
from core.stage_base import Outputs, Stage

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
        model, others, files = self.config.model, self.config.local_models(), self.config.local_files()
        return [super().identity(), directory_signature(Path(model.init)) if model.init else None,
                directory_signature(Path(model.name)), self.config.versions(),
                *([[directory_signature(Path(path)) for path in others]] if others else []),
                *([[sha256_file(Path(path)) if Path(path).is_file() else None for path in files]] if files else [])]

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        checkpoint, summary = workdir / CHECKPOINT, workdir / SUMMARY
        if not (summary.exists() and checkpoint.is_dir()):
            device = resolve_device(self.config.device)
            with self.tracking(workdir) as log:
                if self.config.training.devices > 1:
                    spawn(self.config, inputs, workdir, device, self.progress, log)
                else:
                    tune(self.config, inputs, workdir, self.progress, log, None, device)
        return {"checkpoint": str(checkpoint), "run": str(workdir), **read_json(summary)}

    def history(self, outputs: Outputs) -> History:
        return [(int(event["step"]), tracked(event)) for event in history(Path(outputs["run"]))]


def tune(config: TuningConfig, inputs: dict[str, Outputs], workdir: Path, progress: Progress,
         log: Callable[[int, dict[str, float]], None] | None, world: World | None, device: Device) -> None:
    """Load the backbone, let the method prepare, run the loop: in this process, or as one of `training.devices`.
    With several, the first process prepares (DPO and KTO cache reference values) and the others read its cache."""
    backbone, method, training = config.build_backbone(), config.build_method(), config.training
    progress.update(0, None, "loading model")
    with backbone.loaded(device):
        # Both folds are checked now: a held-out row that does not fit would otherwise fail after training.
        held_out = fitting(backbone, method, read_jsonl(Path(inputs["data"]["validation"])), training, progress, "validation")
        rows = fitting(backbone, method, read_jsonl(Path(inputs["data"]["train"])), training, progress, "training")
        if world is None or world.rank == 0:
            rows = method.prepare(backbone, rows, workdir, progress)
        if world is not None:
            import torch.distributed as distributed
            distributed.barrier()
            if world.rank > 0:
                rows = method.prepare(backbone, rows, workdir, progress)
        backbone.adapt(training.adapter)
        if training.gradient_checkpointing:
            backbone.checkpoint_activations()
        sample, batch_size = held_out[: training.eval_rows], config.validation.batch_size

        def evaluation() -> dict[str, float]:  # the validate stage's measurement, on the weights as they are
            return method.evaluate(backbone, sample, batch_size, progress)

        fit(backbone, method, rows, training, workdir, config.seed, progress, log, evaluation, world)


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

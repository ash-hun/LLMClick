"""Stages every modeling recipe has: training, then a validation gate that must pass before the model is used."""

import os
import math
import logging
from abc import abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from core.stage import Outputs, Stage
from core.utils.files import write_json
from modeling.config import ModelingConfig, ValidationConfig
from modeling.tracker import History, Tracker

ModelingConfigT = TypeVar("ModelingConfigT", bound=ModelingConfig)
logger = logging.getLogger(__name__)


class ValidationFailed(RuntimeError):
    pass


class TrainStage(Stage[ModelingConfigT]):
    """Template: the recipe trains and reports its history; the pipeline sends that history to the tracker.
    Outputs carry `checkpoint`."""
    name: ClassVar[str] = "train"

    @abstractmethod
    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        """Train (or resume) inside `workdir`; return outputs with `checkpoint`, or {} when there is nothing to train."""

    def history(self, outputs: Outputs) -> History:
        return []

    @contextmanager
    def tracking(self, workdir: Path) -> Iterator[Callable[[int, dict[str, float]], None]]:
        """A function that sends one step's metrics to the tracker while training runs. The run is marked as sent
        only if the block ends normally; after a crash the rerun continues the same tracked run."""
        tracker = Tracker(self.config.tracker)
        tracker.open(workdir, self.config.name, self.config.model_dump(mode="json"))
        complete = False
        try:
            yield tracker.log
            complete = True
        finally:
            tracker.close(workdir, complete)

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        return self.train(workdir, inputs)


def violations(metrics: dict[str, float], bounds: ValidationConfig) -> list[str]:
    """Why these metrics fail the bounds. An unmeasured bounded metric is a failure, and so is any metric that is
    not a finite number: NaN compares false against every bound, so it would otherwise pass them all."""
    found = [f"{name} was not measured" for name in sorted({*bounds.min, *bounds.max} - set(metrics))]
    found += [f"{name} is {value}, not a finite number" for name, value in sorted(metrics.items()) if not math.isfinite(value)]
    finite = {name: value for name, value in metrics.items() if math.isfinite(value)}
    found += [f"{name}={finite[name]:.4f} is below the minimum {low}" for name, low in sorted(bounds.min.items())
              if name in finite and finite[name] < low]
    found += [f"{name}={finite[name]:.4f} is above the maximum {high}" for name, high in sorted(bounds.max.items())
              if name in finite and finite[name] > high]
    return found


class ValidateStage(Stage[ModelingConfigT]):
    """Template: the recipe measures the trained model on held-out data; the bounds check and the report are here.
    The checkpoint is passed on only when validation passes: this stage's `checkpoint` link exists only then, and
    it is what later stages and `model.init` of a follow-up experiment should read."""
    name: ClassVar[str] = "validate"
    requires: ClassVar[tuple[str, ...]] = ("train",)
    sections: ClassVar[tuple[str, ...]] = ("validation",)
    REPORT: ClassVar[str] = "validation.json"
    CHECKPOINT: ClassVar[str] = "checkpoint"

    @abstractmethod
    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        """Metrics of the trained model on data it has not seen."""

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        checkpoint = inputs["train"].get("checkpoint")
        if not checkpoint:
            raise RuntimeError("Nothing to validate: no trained checkpoint and no `checkpoint` in the config")
        metrics = self.measure(workdir, inputs)
        failures = violations(metrics, self.config.validation)
        report: dict[str, Any] = {"passed": not failures, "failures": failures, "metrics": metrics,
                                  "checkpoint": checkpoint}
        write_json(workdir / self.REPORT, report)
        link = workdir / self.CHECKPOINT
        if link.is_symlink():
            link.unlink()  # a rerun that fails must not leave an earlier pass's link behind
        if failures:
            raise ValidationFailed(f"Validation failed for {checkpoint}: " + "; ".join(failures))
        if not (self.config.validation.min or self.config.validation.max):
            logger.warning("validation has no bounds (validation.min / validation.max): metrics are recorded, nothing is checked")
        link.symlink_to(os.path.relpath(checkpoint, workdir))
        return {"report": str(workdir / self.REPORT), "passed": True, "metrics": metrics, "checkpoint": str(link)}

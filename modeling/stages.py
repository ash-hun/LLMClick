"""Stages every modeling recipe has: training, then a validation gate that must pass before the model is used."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from core.stage import Outputs, Stage
from core.utils.files import write_json
from modeling.config import ModelingConfig, ValidationConfig
from modeling.tracker import History, Tracker

ModelingConfigT = TypeVar("ModelingConfigT", bound=ModelingConfig)


class ValidationFailed(RuntimeError):
    pass


class TrainStage(Stage[ModelingConfigT]):
    """Template: the recipe trains and reports its history; tracking is handled here. Outputs carry `checkpoint`."""
    name: ClassVar[str] = "train"

    @abstractmethod
    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        """Train (or resume) inside `workdir`; return outputs with `checkpoint`, or {} when there is nothing to train."""

    def history(self, outputs: Outputs) -> History:
        return []

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        outputs = self.train(workdir, inputs)
        Tracker(self.config.tracker).sync(workdir.name, self.config.name, self.history(outputs),
                                          self.config.model_dump(mode="json"))
        return outputs


def violations(metrics: dict[str, float], bounds: ValidationConfig) -> list[str]:
    """Why these metrics fail the bounds; an unmeasured bounded metric is a failure, not a pass."""
    found = [f"{name} was not measured" for name in sorted({*bounds.min, *bounds.max} - set(metrics))]
    found += [f"{name}={metrics[name]:.4f} is below the minimum {low}" for name, low in sorted(bounds.min.items())
              if name in metrics and metrics[name] < low]
    found += [f"{name}={metrics[name]:.4f} is above the maximum {high}" for name, high in sorted(bounds.max.items())
              if name in metrics and metrics[name] > high]
    return found


class ValidateStage(Stage[ModelingConfigT]):
    """Template: the recipe measures the trained model on held-out data; the bounds check and the report are here.
    The checkpoint is passed on only when validation passes, so later stages read it from this stage."""
    name: ClassVar[str] = "validate"
    requires: ClassVar[tuple[str, ...]] = ("train",)
    sections: ClassVar[tuple[str, ...]] = ("validation",)
    REPORT: ClassVar[str] = "validation.json"

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
        if failures:
            raise ValidationFailed(f"Validation failed for {checkpoint}: " + "; ".join(failures))
        return {"report": str(workdir / self.REPORT), "passed": True, "metrics": metrics, "checkpoint": checkpoint}

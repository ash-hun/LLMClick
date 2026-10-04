"""Training Method base: what a row must contain, how a batch becomes a loss, and how the result is measured."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from pydantic import BaseModel

from modeling.tuning.backbone import Backbone
from modeling.tuning.config import TrainingConfig
from core.progress import Progress

Row = dict[str, Any]
BackboneT = TypeVar("BackboneT", bound=Backbone)


def chunks(rows: list[Row], size: int) -> list[list[Row]]:
    return [rows[start:start + size] for start in range(0, len(rows), size)]


class TrainingMethod(ABC, Generic[BackboneT]):
    class Config(BaseModel):
        """Method-specific keys of the `method:` section; subclasses replace it."""

    def __init__(self, config: Any, training: TrainingConfig) -> None:
        self.config = config
        self.training = training
        self.metrics: dict[str, float] = {}  # extra numbers of the last `loss` call, logged next to the loss

    @abstractmethod
    def check(self, row: Row) -> None:
        """Raise ValueError when the row lacks what this method trains on."""

    def prepare(self, backbone: BackboneT, rows: list[Row], workdir: Path) -> list[Row]:
        """Once before training, with the untouched base weights; may cache into `workdir` and annotate rows."""
        return rows

    def parameter_groups(self, backbone: BackboneT) -> list[dict[str, Any]]:
        """Optimizer groups; a group may carry its own `lr`, which follows the same warmup and decay."""
        return [{"params": backbone.trainable()}]

    @abstractmethod
    def loss(self, backbone: BackboneT, rows: list[Row]) -> torch.Tensor:
        """Differentiable scalar for one batch of rows."""

    def finish(self, backbone: BackboneT) -> dict[str, float]:
        """Once after the last step and before the checkpoint is written, e.g. to fit a calibration temperature."""
        return {}

    @abstractmethod
    def evaluate(self, backbone: BackboneT, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """Metrics on held-out rows; the validation bounds in the config name these keys."""

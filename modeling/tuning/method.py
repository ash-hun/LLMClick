"""Training Method base: what a row must contain, how a batch becomes a loss, and how the result is measured."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import torch

from modeling.tuning.backbone import Backbone
from modeling.tuning.config import TrainingConfig
from core.config.schema import Section
from core.progress import Progress

Row = dict[str, Any]
logger = logging.getLogger(__name__)
BackboneT = TypeVar("BackboneT", bound=Backbone)


def chunks(rows: list[Row], size: int) -> list[list[Row]]:
    return [rows[start:start + size] for start in range(0, len(rows), size)]


class TrainingMethod(ABC, Generic[BackboneT]):
    version: ClassVar[int] = 1  # bump when this method's code changes what the same config trains or measures

    class Config(Section):
        """Method-specific keys of the `method:` section; subclasses replace it."""

    def __init__(self, config: Any, training: TrainingConfig) -> None:
        self.config = config
        self.training = training
        self.metrics: dict[str, float] = {}  # extra numbers of the last `loss` call, logged next to the loss

    @abstractmethod
    def check(self, row: Row) -> None:
        """Raise ValueError when the row lacks what this method trains on."""

    def check_identity(self) -> Any:
        """The part of the method's config that `check` reads; it joins the data stage's fingerprint, so changing
        it checks the rows again instead of reusing rows checked under the old setting."""
        return None

    def lengths(self, backbone: BackboneT, row: Row) -> list[int]:
        """Token counts of the sequences this row becomes, at their longest; they are checked against
        `training.max_length` before any training time is spent. Empty when the method may cut text safely."""
        return []

    def prepare(self, backbone: BackboneT, rows: list[Row], workdir: Path, progress: Progress) -> list[Row]:
        """Once before training, with the untouched base weights; may cache into `workdir` and annotate rows."""
        return rows

    def parameter_groups(self, backbone: BackboneT) -> list[dict[str, Any]]:
        """Optimizer groups; a group may carry its own `lr`, which follows the same warmup and decay."""
        return [{"params": backbone.trainable()}]

    @abstractmethod
    def loss(self, backbone: BackboneT, rows: list[Row]) -> torch.Tensor:
        """Differentiable scalar for one batch of rows."""

    def finish(self, backbone: BackboneT, progress: Progress) -> dict[str, float]:
        """Once after the last step and before the checkpoint is written, e.g. to fit a calibration temperature."""
        return {}

    @abstractmethod
    def evaluate(self, backbone: BackboneT, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """Metrics on held-out rows; the validation bounds in the config name these keys."""


def fitting(backbone: Backbone, method: TrainingMethod[Any], rows: list[Row], training: TrainingConfig, progress: Progress,
            name: str) -> list[Row]:
    """The rows whose sequences fit `training.max_length`. With `overflow: error` any other row stops the run here,
    before training; with `skip` such rows are left out. Either way nothing is cut and nothing fails hours later."""
    kept: list[tuple[int, int, Row]] = []
    over: list[tuple[int, int, Row]] = []
    for index, row in enumerate(rows):
        if index % 256 == 0:
            progress.update(index, len(rows), f"checking {name} lengths")
        longest = max(method.lengths(backbone, row), default=0)
        (over if longest > training.max_length else kept).append((index, longest, row))
    progress.update(len(rows), len(rows), f"checking {name} lengths")
    if over and (training.overflow == "error" or not kept):
        worst = ", ".join(f"row {index}: {longest}" for index, longest, _ in sorted(over, key=lambda item: -item[1])[:5])
        raise ValueError(f"{len(over)} of {len(rows)} {name} rows need more than training.max_length={training.max_length} "
                         f"tokens ({worst}). Raise max_length, shorten the rows, or set training.overflow: skip")
    if over:
        logger.warning("%d of %d %s rows exceed max_length=%d and are left out", len(over), len(rows), name, training.max_length)
    return [row for _, _, row in kept]

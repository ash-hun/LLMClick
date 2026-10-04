"""Model Catalog base: one architecture family that knows how to load, save and expose its trainable weights."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from transformers.utils import logging as transformers_logging

from modeling.tuning.config import BackboneConfig
from core.utils.device import Device

# transformers' weight-loading bars would draw over the pipeline's own
transformers_logging.disable_progress_bar()  # type: ignore[no-untyped-call]


class Backbone(ABC):
    model: Any
    tokenizer: Any
    device: Device

    def __init__(self, config: BackboneConfig) -> None:
        self.config = config

    def origin(self, checkpoint: Path | None) -> tuple[str, str | None]:
        """Where weights come from: a checkpoint this pipeline wrote, a local directory, or a pinned Hub revision."""
        if checkpoint is not None:
            return str(checkpoint), None
        return self.config.name, None if Path(self.config.name).is_dir() else self.config.revision

    @abstractmethod
    def load(self, device: Device, checkpoint: Path | None = None) -> None:
        """Set `model`, `tokenizer` and `device` from the base model, or from `checkpoint` when given."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write everything `load(checkpoint=path)` needs."""

    def trainable(self) -> list[torch.nn.Parameter]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def release(self) -> None:
        """Drop the weights so the next stage or job gets the accelerator memory back."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

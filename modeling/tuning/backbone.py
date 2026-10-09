"""Model Catalog base: one architecture family that knows how to load, save and expose its trainable weights."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import torch
from transformers.utils import logging as transformers_logging

from modeling.tuning.config import AdapterConfig, BackboneConfig
from core.utils.device import Device

# transformers' weight-loading bars would draw over the pipeline's own
transformers_logging.disable_progress_bar()  # type: ignore[no-untyped-call]


class Backbone(ABC):
    # Layers LoRA adapts unless the config names others: the attention projections of a standard transformer block.
    adapter_targets: ClassVar[tuple[str, ...]] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # The weights train in FP32: at fine-tuning learning rates a bf16 weight is too coarse to register an update.
    # The one place the dtype is decided; a catalog entry that loads quantized or bf16 weights overrides `load`.
    dtype: ClassVar[torch.dtype] = torch.float32
    version: ClassVar[int] = 1  # bump when this class's code changes what the same config trains
    model: Any
    tokenizer: Any
    device: Device

    def __init__(self, config: BackboneConfig) -> None:
        self.config = config

    def origin(self, checkpoint: Path | None) -> tuple[str, str | None]:
        """Where weights come from: a checkpoint this pipeline wrote, `model.init`, a local directory, or a pinned
        Hub revision."""
        if checkpoint is not None:
            return str(checkpoint), None
        if self.config.init is not None:
            # checked here, not in the config: a config that continues another experiment must validate before
            # that experiment has run
            if not Path(self.config.init).is_dir():
                raise FileNotFoundError(f"model.init {self.config.init!r} does not exist; run the experiment that writes it")
            return self.config.init, None
        return self.config.name, None if Path(self.config.name).is_dir() else self.config.revision

    @abstractmethod
    def load(self, device: Device, checkpoint: Path | None = None) -> None:
        """Set `model`, `tokenizer` and `device` from the base model, or from `checkpoint` when given."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write everything `load(checkpoint=path)` needs."""

    def adapt(self, adapter: AdapterConfig | None) -> None:
        """Wrap the loaded model so that only adapter weights train; without an adapter every weight trains."""
        if adapter is None:
            return
        from peft import LoraConfig, get_peft_model
        self.model = get_peft_model(self.model, LoraConfig(
            r=adapter.r, lora_alpha=adapter.alpha, lora_dropout=adapter.dropout,
            target_modules=list(adapter.targets or self.adapter_targets)))

    def exported(self) -> Any:
        """The model to save: adapter weights folded into the base, so a checkpoint loads like any other model."""
        return self.model.merge_and_unload() if hasattr(self.model, "merge_and_unload") else self.model

    def modules(self) -> dict[str, torch.nn.Module]:
        """Everything that holds trainable weights, by name; a backbone with a head adds it here."""
        return {"model": self.model}

    def trainable(self) -> list[torch.nn.Parameter]:
        return [parameter for module in self.modules().values() for parameter in module.parameters()
                if parameter.requires_grad]

    def snapshot(self) -> dict[str, torch.Tensor]:
        """The weights that train, for a resume file: nothing frozen is written."""
        return {f"{prefix}.{name}": parameter for prefix, module in self.modules().items()
                for name, parameter in module.named_parameters() if parameter.requires_grad}

    def restore(self, weights: dict[str, torch.Tensor]) -> None:
        for prefix, module in self.modules().items():
            own = {name.removeprefix(f"{prefix}."): value for name, value in weights.items() if name.startswith(f"{prefix}.")}
            module.load_state_dict(own, strict=False)

    def release(self) -> None:
        """Drop the weights so the next stage or job gets the accelerator memory back."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

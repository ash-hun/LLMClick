"""Run a Qwen3.5 Jeff checkpoint with MLX, Apple's framework, which has fast Metal kernels for Qwen3.5's Gated DeltaNet
layers (PyTorch on a Mac GPU only has a slow reference version of them).

The checkpoint is used as saved; nothing is converted on disk. MLX loads the backbone weights (renamed to the layout its
Qwen3.5 loader expects; the vision tower is left out, so this path is text only) and runs the model to its final
hidden state. Our trained answer readout, answer codes and fitted temperature are then applied exactly as in
jeff.model, and the prompt is built by the same function with the checkpoint's own chat template."""

import json
import math
from collections.abc import Sequence
from pathlib import Path

from jeff.model import MAX_OPTIONS, PROMPT_LAYOUTS, decision_messages, options
from jeff.types import DecisionInput


class MlxDecisionModel:
    backend = "mlx"

    def __init__(self, checkpoint: str | Path) -> None:
        import mlx.core as mx
        from mlx_lm.models.qwen3_5 import Model, ModelArgs
        from transformers import AutoProcessor

        directory = Path(checkpoint)
        decision = json.loads((directory / "decision_config.json").read_text())
        # Qwen3.5 checkpoints (jeff.model) record no architecture; Gemma and ModernBERT checkpoints name theirs.
        if decision.get("format_version") != 1 or "architecture" in decision:
            raise ValueError(f"{directory} is not a Qwen3.5 Jeff checkpoint")
        config = json.loads((directory / "config.json").read_text())
        if config.get("model_type") != "qwen3_5":
            raise ValueError(f"The MLX backend serves Qwen3.5 checkpoints; {directory} is {config.get('model_type')!r}")
        self.base_model = str(decision["base_model"])
        self.codes: list[str] = list(decision["codes"])
        self.temperature = float(decision["temperature"])
        # Checkpoints made before prompt layouts existed use the original state-first order.
        self.prompt_layout = str(decision.get("prompt_layout", "state-first"))
        if self.prompt_layout not in PROMPT_LAYOUTS:
            raise ValueError(f"{directory} has an unknown prompt layout {self.prompt_layout!r}")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be positive and finite.")
        self.mx = mx
        self.model = Model(ModelArgs.from_dict(config))
        raw = mx.load(str(directory / "model.safetensors"))
        # Our checkpoints store the bare Qwen3.5 model ("language_model.*", "visual.*"); MLX's loader expects the
        # Hugging Face full-model names ("model.language_model.*") and applies its own conversions from there.
        renamed = {f"model.{name}": value for name, value in raw.items() if name.startswith("language_model.")}
        if not renamed:
            raise ValueError(f"{directory} has no language_model weights")
        self.model.load_weights(list(self.model.sanitize(renamed).items()), strict=True)
        self.model.eval()  # MLX modules start in training mode, which makes Qwen3.5 skip its fast Metal kernel
        self.readout = mx.load(str(directory / "readout.safetensors"))["weight"].astype(mx.float32)
        if self.readout.shape[0] != MAX_OPTIONS:
            raise ValueError(f"Readout has {self.readout.shape[0]} rows, expected {MAX_OPTIONS}")
        mx.eval(self.model.parameters(), self.readout)
        self.processor = AutoProcessor.from_pretrained(str(directory))

    def prompt_ids(self, row: DecisionInput) -> list[int]:
        if row.get("images"):
            raise ValueError("The MLX backend reads text only; use the PyTorch backend for image decisions")
        text = self.processor.apply_chat_template(decision_messages(row, self.codes, self.prompt_layout), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        return list(self.processor.tokenizer(text, add_special_tokens=False)["input_ids"])

    def decide(self, rows: Sequence[DecisionInput]) -> list[tuple[list[float], int]]:
        """Probabilities over each row's options (temperature applied) and the number of input tokens, one row at a time."""
        mx = self.mx
        results = []
        for row in rows:
            count = len(options(row["question"])[0])
            ids = self.prompt_ids(row)
            hidden = self.model.language_model.model(mx.array([ids]))[0, -1].astype(mx.float32)
            logits = (self.readout[:count] @ hidden) / self.temperature
            probabilities = mx.softmax(logits, axis=-1)
            results.append(([float(p) for p in probabilities.tolist()], len(ids)))
        return results

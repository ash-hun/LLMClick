"""An encoder (ModernBERT) decision model with the same interface as the decoder DecisionModel.

Each option is scored as a cross-encoder pair: (task instructions + state, option description). The first-token
representation goes through a linear scorer, giving one logit per option, laid out exactly like the decoder's readout
(up to 255 options, padding masked out) so training, calibration, evaluation and serving are unchanged."""

import json
import math
import string
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer

from jeff.model import MAX_OPTIONS, PreparedBatch, answer, describe, options
from jeff.types import Answer, DecisionInput, JSONValue

ENCODER_MODELS = {"answerdotai/ModernBERT-base": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
                  "answerdotai/ModernBERT-large": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"}


def first_segment(row: DecisionInput) -> str:
    question = row["question"]
    return f"Question: {describe(question.get('instructions') or 'Choose the best matching option.')}\n\nState:\n{describe(row['state'])}"


class EncoderDecisionModel(torch.nn.Module):
    architecture = "encoder"

    def __init__(
        self, checkpoint: str | Path | None = None, train: bool = False, device: str | None = None,
        *, base_model: str | None = None, revision: str | None = None, gradient_checkpointing: bool = False,
        cpu_threads: int = 8, cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        torch.set_num_threads(cpu_threads)
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        saved: dict[str, JSONValue] | None = None
        if checkpoint is not None:
            saved = json.loads((Path(checkpoint) / "decision_config.json").read_text())
            if saved["format_version"] != 1 or saved.get("architecture") != self.architecture:
                raise ValueError(f"{checkpoint} is not an encoder decision checkpoint")
        elif base_model is None or revision is None:
            raise ValueError("An encoder model needs a checkpoint, or a base_model and revision")
        self.base_model = str(saved["base_model"]) if saved else str(base_model)
        self.revision = str(saved["revision"]) if saved else str(revision)
        if len(self.revision) != 40 or any(character not in string.hexdigits for character in self.revision):
            raise ValueError("Use an immutable, 40-character model revision.")
        source = str(checkpoint) if checkpoint else self.base_model
        pinned = None if checkpoint or Path(self.base_model).is_dir() else self.revision
        cache = str(cache_dir) if cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=pinned, cache_dir=cache)
        dtype = torch.bfloat16 if self.device_name.startswith(("cuda", "mps")) else torch.float32  # half precision on GPUs
        self.backbone = AutoModel.from_pretrained(source, revision=pinned, cache_dir=cache, dtype=dtype, attn_implementation="sdpa")
        self.scorer = torch.nn.Linear(self.backbone.config.hidden_size, 1, dtype=dtype)
        if saved:
            self.scorer.load_state_dict(load_file(str(Path(checkpoint) / "scorer.safetensors")))  # type: ignore[arg-type]
        self.requires_grad_(train)
        if train and gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.to(self.device_name)
        self.temperature = float(saved["temperature"]) if saved else 1.0  # type: ignore[arg-type]
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be positive and finite.")
        self.train(train)

    def prepare(self, rows: Sequence[DecisionInput], max_length: int = 8192) -> PreparedBatch:
        if not rows:
            raise ValueError("A batch must contain at least one decision.")
        firsts: list[str] = []
        seconds: list[str] = []
        counts: list[int] = []
        for row in rows:
            if row.get("images"):
                raise ValueError("The encoder decision model reads text only")
            keys, descriptions = options(row["question"])
            if not 1 <= len(keys) <= MAX_OPTIONS:
                raise ValueError("Questions must have 1 to 255 options.")
            counts.append(len(keys))
            for key, description in zip(keys, descriptions, strict=True):
                firsts.append(first_segment(row))
                seconds.append(f"Option {key}: {describe(description)}")
        encoded = self.tokenizer(firsts, seconds, padding=True, return_tensors="pt")
        if encoded["input_ids"].shape[1] > max_length:
            raise ValueError(f"Question branch exceeds the {max_length}-token limit; no input was truncated.")
        inputs = {name: tensor.to(self.device_name) for name, tensor in encoded.items() if name in ("input_ids", "attention_mask")}
        return PreparedBatch(inputs, tuple(counts), int(encoded["attention_mask"].sum()))

    def forward(self, batch: PreparedBatch) -> torch.Tensor:
        hidden: torch.Tensor = self.backbone(**batch.inputs).last_hidden_state[:, 0]
        scores = self.scorer(hidden).float().squeeze(-1)
        logits = torch.full((len(batch.counts), MAX_OPTIONS), -1e9, device=scores.device)
        offset = 0
        for index, count in enumerate(batch.counts):
            logits[index, :count] = scores[offset:offset + count]
            offset += count
        return logits

    @torch.inference_mode()
    def predict(self, rows: Sequence[DecisionInput], batch_size: int = 8, temperature: float | None = None) -> list[list[float]]:
        scale = self.temperature if temperature is None else temperature
        if not math.isfinite(scale) or scale <= 0 or batch_size < 1:
            raise ValueError("Temperature and batch size must be positive.")
        was_training = self.training
        self.eval()
        distributions: list[list[float]] = []
        try:
            for start in range(0, len(rows), batch_size):
                batch = self.prepare(rows[start:start + batch_size])
                probabilities: list[list[float]] = (self(batch) / scale).softmax(-1).cpu().tolist()
                distributions.extend(values[:count] for values, count in zip(probabilities, batch.counts))
        finally:
            self.train(was_training)
        return distributions

    def decide(self, rows: Sequence[DecisionInput], batch_size: int = 8) -> list[Answer]:
        return [answer(row["question"], probabilities) for row, probabilities in zip(rows, self.predict(rows, batch_size))]

    def save(self, directory: str | Path, temperature: float | None = None, **metadata: JSONValue) -> None:
        destination = Path(directory)
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Refusing to overwrite checkpoint contents: {destination}")
        scale = self.temperature if temperature is None else temperature
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Temperature must be positive and finite.")
        destination.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(destination))
        self.tokenizer.save_pretrained(str(destination))
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in self.scorer.state_dict().items()},
                  str(destination / "scorer.safetensors"))
        config: dict[str, JSONValue] = dict(metadata)
        config.update({"format_version": 1, "architecture": self.architecture, "base_model": self.base_model,
                       "revision": self.revision, "temperature": scale})
        (destination / "decision_config.json").write_text(json.dumps(config, indent=2) + "\n")

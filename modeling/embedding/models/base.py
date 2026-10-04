"""What every embedding backbone offers the training methods: texts in, unit vectors out."""

from abc import abstractmethod
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from modeling.tuning.backbone import Backbone
from core.utils.device import Device


class EmbeddingBackbone(Backbone):
    def load(self, device: Device, checkpoint: Path | None = None) -> None:
        origin, revision = self.origin(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(origin, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(origin, revision=revision, dtype=torch.float32).to(device)
        self.device = device

    def save(self, path: Path) -> None:
        self.exported().save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @abstractmethod
    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """One vector per sequence from the token states."""

    def embed(self, texts: list[str], max_length: int) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        hidden = self.model(**batch).last_hidden_state
        return torch.nn.functional.normalize(self.pool(hidden, batch["attention_mask"]).float(), dim=-1)

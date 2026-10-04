"""What every LLM backbone offers the training methods: chat rendering, token ids, logits and generation."""

from pathlib import Path
from typing import Any, ClassVar

import torch
from transformers import AutoTokenizer

from modeling.tuning.backbone import Backbone
from core.utils.device import Device

Message = dict[str, str]


class LLMBackbone(Backbone):
    loader: ClassVar[Any]                        # the transformers Auto class that builds this architecture
    frozen: ClassVar[tuple[str, ...]] = ()       # parameter-name fragments that text training must not update

    def load(self, device: Device, checkpoint: Path | None = None) -> None:
        origin, revision = self.origin(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(origin, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # FP32 weights: at fine-tuning learning rates a bf16 weight is too coarse to register an update.
        self.model = self.loader.from_pretrained(origin, revision=revision, dtype=torch.float32).to(device)
        for name, parameter in self.model.named_parameters():
            if any(fragment in name for fragment in self.frozen):
                parameter.requires_grad_(False)
        self.device = device

    def save(self, path: Path) -> None:
        self.exported().save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def render(self, messages: list[Message], generation_prompt: bool) -> list[int]:
        """Token ids of a conversation in the model's own chat format."""
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt,
                                                  **self.config.template)
        ids: list[int] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids

    def padded(self, sequences: list[list[int]], left: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(sequence) for sequence in sequences)
        ids = torch.full((len(sequences), width), int(self.tokenizer.pad_token_id), dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            span = slice(width - len(sequence), width) if left else slice(0, len(sequence))
            ids[row, span] = torch.tensor(sequence, dtype=torch.long)
            mask[row, span] = 1
        return ids.to(self.device), mask.to(self.device)

    def logits(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-token logits for right-padded sequences, with their attention mask."""
        ids, mask = self.padded(sequences)
        logits: torch.Tensor = self.model(input_ids=ids, attention_mask=mask).logits
        return logits, mask

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, samples: int = 1,
                 temperature: float = 0.0) -> list[list[list[int]]]:
        """Per prompt, `samples` completions as token ids cut at the end token; temperature 0 decodes greedily."""
        ids, mask = self.padded(prompts, left=True)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        training = self.model.training
        self.model.eval()
        output = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                     num_return_sequences=samples, pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=self.tokenizer.eos_token_id, **sampling)
        self.model.train(training)
        completions = output[:, ids.shape[1]:].tolist()
        stop = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        cut = [next((tokens[:i] for i, token in enumerate(tokens) if token in stop), tokens) for tokens in completions]
        return [cut[start:start + samples] for start in range(0, len(cut), samples)]

    def text(self, tokens: list[int]) -> str:
        decoded: str = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded

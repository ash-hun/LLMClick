"""What every LLM backbone offers the training methods: chat rendering, token ids, logits and generation."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from modeling.tuning.backbone import Backbone
from core.utils.device import Device
from core.utils.files import write_json

if TYPE_CHECKING:
    from modeling.llm.models.heads.base import DecisionHead

Message = dict[str, str]
THINK, THINK_END = "<think>", "</think>"
HEAD_WEIGHTS, HEAD_CONFIG = "head.safetensors", "head.json"


class LLMBackbone(Backbone):
    loader: ClassVar[Any]                        # the transformers Auto class that builds this architecture
    frozen: ClassVar[tuple[str, ...]] = ()       # parameter-name fragments that text training must not update
    head: "DecisionHead | None" = None           # set when `model.head` names one

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
        if self.config.head is not None:
            self.load_head(Path(origin))

    def load_head(self, origin: Path) -> None:
        """A fresh head on a base model; the saved head (weights, option codes, temperature) on a checkpoint."""
        from modeling.llm.models.heads import HEADS
        saved = json.loads((origin / HEAD_CONFIG).read_text()) if (origin / HEAD_CONFIG).exists() else None
        if saved is not None and saved["head"] != self.config.head:
            raise ValueError(f"{origin} was trained with head {saved['head']!r}, the config asks for {self.config.head!r}")
        self.head = HEADS.get(str(self.config.head))(self, saved)
        assert self.head is not None
        if saved is not None:
            self.head.load_state_dict(load_file(str(origin / HEAD_WEIGHTS)))
        self.head.to(self.device)

    def modules(self) -> dict[str, torch.nn.Module]:
        return {**super().modules(), **({"head": self.head} if self.head is not None else {})}

    def save(self, path: Path) -> None:
        self.exported().save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.head is not None:
            save_file({name: value.detach().cpu().contiguous() for name, value in self.head.state_dict().items()},
                      str(path / HEAD_WEIGHTS))
            write_json(path / HEAD_CONFIG, {"head": self.config.head, **self.head.settings()})

    def release(self) -> None:
        self.head = None
        super().release()

    def token_ids(self, text: str) -> list[int]:
        ids: list[int] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids

    def think_prompt(self, content: str) -> list[int]:
        """One user turn followed by an open reasoning block. Qwen3.5's template opens the block itself; Qwen3's
        leaves that to the model, so it is added here."""
        text: str = self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=True)
        return self.token_ids(text if text.endswith(f"{THINK}\n") else f"{text}{THINK}\n")

    def hidden(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Final hidden states of right-padded sequences, with their token ids and attention mask. The language-model
        head is not applied: over a whole batch its output (tokens x vocabulary) is by far the largest tensor."""
        ids, mask = self.padded(sequences)
        inner = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model  # under LoRA
        states: torch.Tensor = inner.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
        return states, ids, mask

    def next_token_log_probabilities(self, hidden: torch.Tensor, ids: torch.Tensor, graded: torch.Tensor,
                                     temperature: float = 1.0) -> torch.Tensor:
        """log p(token | everything before it) for the tokens `graded` marks, zero elsewhere; shape (rows, tokens - 1).
        The language-model head runs only on the marked positions."""
        selected = graded[:, 1:].bool()
        out = hidden.new_zeros(selected.shape, dtype=torch.float32)
        if bool(selected.any()):
            logits = self.model.get_output_embeddings()(hidden[:, :-1][selected]).float() / temperature
            out[selected] = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:][selected].unsqueeze(-1)).squeeze(-1)
        return out

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

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, samples: int = 1, temperature: float = 0.0,
                 stop: int | None = None, suppress: list[int] | None = None) -> list[list[list[int]]]:
        """Per prompt, `samples` completions as token ids cut at the end token (or at `stop`); temperature 0 decodes
        greedily. `suppress` lists tokens that may never be produced."""
        ids, mask = self.padded(prompts, left=True)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        training = self.model.training
        self.model.eval()
        output = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                     num_return_sequences=samples, pad_token_id=self.tokenizer.pad_token_id,
                                     eos_token_id=[self.tokenizer.eos_token_id, *([stop] if stop is not None else [])],
                                     suppress_tokens=suppress, **sampling)
        self.model.train(training)
        completions = output[:, ids.shape[1]:].tolist()
        ends = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, stop}
        cut = [next((tokens[:i] for i, token in enumerate(tokens) if token in ends), tokens) for tokens in completions]
        return [cut[start:start + samples] for start in range(0, len(cut), samples)]

    def text(self, tokens: list[int]) -> str:
        decoded: str = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return decoded

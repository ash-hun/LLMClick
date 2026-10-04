"""A decision head turns the backbone's hidden states into one score per option, and owns the prompt it reads."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from modeling.llm.models.base import THINK_END

if TYPE_CHECKING:
    from modeling.llm.models.base import LLMBackbone

# Rare tokens of the Qwen tokenizer reused as structure markers; plain words in their place score worse (Jeeves).
STATE, QUESTION, OPTION, OPTION_END, DECIDE = "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"
CONTROL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
MASKED = -1e9  # finite, so a zero target times a masked score stays zero


def sanitized(text: str) -> str:
    """User text must not contain a control token, or it could forge a marker the head reads."""
    return CONTROL.sub(r"<¦\1¦>", text)


@dataclass
class Layout:
    """One question as the model reads it: optional reasoning, then the options again and the decide marker."""
    ids: list[int]
    reasoning: tuple[int, int]   # token span of the reasoning chain (and its closing token when it closed)
    options: int


class DecisionHead(torch.nn.Module, ABC):
    name: ClassVar[str]

    def __init__(self, backbone: "LLMBackbone", saved: dict[str, Any] | None) -> None:
        super().__init__()
        markers = [STATE, QUESTION, OPTION, OPTION_END, DECIDE, THINK_END]
        ids = backbone.tokenizer.convert_tokens_to_ids(markers)
        missing = [marker for marker, i in zip(markers, ids) if i is None or i == backbone.tokenizer.unk_token_id]
        if missing:
            raise ValueError(f"tokenizer of {backbone.config.name} lacks the marker tokens {missing}")
        self.option_end, self.think_end = int(ids[3]), int(ids[5])
        self.temperature = float(saved["temperature"]) if saved else 1.0
        self.hidden_size = int(backbone.model.get_input_embeddings().embedding_dim)

    def settings(self) -> dict[str, Any]:
        """What besides the weights a checkpoint must keep."""
        return {"temperature": self.temperature}

    @abstractmethod
    def option_block(self, options: list[str]) -> str:
        """The options as this head needs to read them."""

    def prompt(self, backbone: "LLMBackbone", state: str, instructions: str, options: list[str]) -> list[int]:
        block = self.option_block([sanitized(option) for option in options])
        return backbone.think_prompt(f"{STATE}{sanitized(state)}\n{QUESTION}{sanitized(instructions)}\n{block}")

    def layout(self, backbone: "LLMBackbone", state: str, instructions: str, options: list[str],
               chain: list[int] | None = None, closed: bool = True) -> Layout:
        """`chain` is the reasoning to insert (None: an empty block); `closed` says whether it ended by itself."""
        prompt = self.prompt(backbone, state, instructions, options)
        body = backbone.token_ids("\n") if chain is None else chain
        block = self.option_block([sanitized(option) for option in options])
        remainder = backbone.token_ids(f"{THINK_END}\n\n{block}{DECIDE}")
        end = len(prompt) + len(body) + int(closed and chain is not None)
        return Layout(prompt + body + remainder, (len(prompt), end), len(options))

    @abstractmethod
    def scores(self, hidden: torch.Tensor, layouts: list[Layout]) -> torch.Tensor:
        """(rows, most options) scores from (rows, tokens, hidden) states; missing options hold MASKED."""

    def probabilities(self, scores: torch.Tensor) -> torch.Tensor:
        """Calibrated: the fitted temperature is applied here, never during training."""
        return torch.softmax(scores / self.temperature, dim=-1)

"""A decision head turns the backbone's hidden states into one score per option, and owns the prompt it reads."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from modeling.llm.models.base import LLMBackbone

MASKED = -1e9  # finite, so a zero target times a masked score stays zero


@dataclass
class Layout:
    """One question as the model reads it: optional reasoning, then the options again and the decide marker."""
    ids: list[int]
    reasoning: tuple[int, int]   # token span of the reasoning chain (and its closing token when it closed)
    options: int


class DecisionHead(torch.nn.Module, ABC):
    name: ClassVar[str]
    version: ClassVar[int] = 1  # bump when the prompt or the scoring changes what the same weights answer

    def __init__(self, backbone: "LLMBackbone", saved: dict[str, Any] | None) -> None:
        super().__init__()
        self.markers = backbone.decision_markers()
        names = [self.markers.state, self.markers.question, self.markers.option, self.markers.option_end,
                 self.markers.decide, self.markers.think_end]
        ids = backbone.tokenizer.convert_tokens_to_ids(names)
        missing = [name for name, i in zip(names, ids, strict=True) if i is None or i == backbone.tokenizer.unk_token_id]
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
        block = self.option_block([backbone.neutral(option) for option in options])
        return backbone.think_prompt(f"{self.markers.state}{backbone.neutral(state)}\n"
                                     f"{self.markers.question}{backbone.neutral(instructions)}\n{block}")

    def layout(self, backbone: "LLMBackbone", state: str, instructions: str, options: list[str],
               chain: list[int] | None = None, closed: bool = True) -> Layout:
        """`chain` is the reasoning to insert (None: an empty block); `closed` says whether it ended by itself."""
        prompt = self.prompt(backbone, state, instructions, options)
        body = backbone.token_ids("\n") if chain is None else chain
        block = self.option_block([backbone.neutral(option) for option in options])
        remainder = backbone.token_ids(f"{self.markers.think_end}\n\n{block}{self.markers.decide}")
        end = len(prompt) + len(body) + int(closed and chain is not None)
        return Layout(prompt + body + remainder, (len(prompt), end), len(options))

    @abstractmethod
    def scores(self, hidden: torch.Tensor, layouts: list[Layout]) -> torch.Tensor:
        """(rows, most options) scores from (rows, tokens, hidden) states; missing options hold MASKED."""

    def probabilities(self, scores: torch.Tensor) -> torch.Tensor:
        """Calibrated: the fitted temperature is applied here, never during training."""
        return torch.softmax(scores / self.temperature, dim=-1)

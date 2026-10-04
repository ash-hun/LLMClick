"""Readout head (Jev, jeff): options get one-token codes, and a linear layer on the decide position scores the codes."""

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from modeling.llm.models.heads.base import MASKED, DecisionHead, Layout
from modeling.llm.models.heads import HEADS

if TYPE_CHECKING:
    from modeling.llm.models.base import LLMBackbone

MAX_OPTIONS = 255
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def single_token_codes(backbone: "LLMBackbone") -> list[str]:
    """A, B, ... then AA, AB, ...: the first MAX_OPTIONS candidates the tokenizer keeps as one token."""
    candidates = [*LETTERS, *(a + b for a in LETTERS for b in LETTERS)]
    return [code for code in candidates if len(backbone.token_ids(code)) == 1][:MAX_OPTIONS]


@HEADS.register("readout")
class ReadoutHead(DecisionHead):
    """One forward pass, one linear layer; starts from the language-model head's rows for the code tokens."""
    name: ClassVar[str] = "readout"

    def __init__(self, backbone: "LLMBackbone", saved: dict[str, Any] | None) -> None:
        super().__init__(backbone, saved)
        self.codes: list[str] = saved["codes"] if saved else single_token_codes(backbone)
        self.linear = torch.nn.Linear(self.hidden_size, len(self.codes), bias=False)
        if saved is None:
            rows = [backbone.token_ids(code)[0] for code in self.codes]
            with torch.no_grad():
                self.linear.weight.copy_(backbone.model.get_output_embeddings().weight[rows].float())

    def settings(self) -> dict[str, Any]:
        return {**super().settings(), "codes": self.codes}

    def option_block(self, options: list[str]) -> str:
        if len(options) > len(self.codes):
            raise ValueError(f"{len(options)} options, but this tokenizer yields only {len(self.codes)} one-token codes")
        return "".join(f"{code}: {option}\n" for code, option in zip(self.codes, options))

    def scores(self, hidden: torch.Tensor, layouts: list[Layout]) -> torch.Tensor:
        last = torch.tensor([len(layout.ids) - 1 for layout in layouts], device=hidden.device)
        decide = hidden[torch.arange(len(layouts), device=hidden.device), last]
        widest = max(layout.options for layout in layouts)
        scores = self.linear(decide.to(self.linear.weight.dtype)).float()[:, :widest]
        counts = torch.tensor([layout.options for layout in layouts], device=hidden.device)
        masked: torch.Tensor = scores.masked_fill(torch.arange(widest, device=hidden.device)[None] >= counts[:, None], MASKED)
        return masked

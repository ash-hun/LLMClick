"""Pointer head (Jeeves): the decide position asks, each option's end position answers, a scaled dot product scores."""

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from modeling.llm.models.heads.base import MASKED, OPTION, OPTION_END, DecisionHead, Layout
from modeling.llm.models.heads import HEADS

if TYPE_CHECKING:
    from modeling.llm.models.base import LLMBackbone

LATENT = 256


@HEADS.register("pointer")
class PointerHead(DecisionHead):
    """Any number of options, no option codes: the head reads the option texts' own representations."""
    name: ClassVar[str] = "pointer"

    def __init__(self, backbone: "LLMBackbone", saved: dict[str, Any] | None) -> None:
        super().__init__(backbone, saved)
        self.query = torch.nn.Linear(self.hidden_size, LATENT)
        self.key = torch.nn.Linear(self.hidden_size, LATENT)

    def option_block(self, options: list[str]) -> str:
        return "".join(f"{OPTION}{option}{OPTION_END}\n" for option in options)

    def scores(self, hidden: torch.Tensor, layouts: list[Layout]) -> torch.Tensor:
        out = hidden.new_full((len(layouts), max(layout.options for layout in layouts)), MASKED, dtype=torch.float32)
        for row, layout in enumerate(layouts):
            # only the option block repeated after the reasoning is read, so the reasoning can change the answer
            ends = [i for i in range(layout.reasoning[1], len(layout.ids)) if layout.ids[i] == self.option_end]
            if len(ends) != layout.options:
                raise ValueError(f"expected {layout.options} option boundaries after the reasoning, found {len(ends)}")
            keys = self.key(hidden[row, ends].to(self.key.weight.dtype))
            query = self.query(hidden[row, len(layout.ids) - 1].to(self.query.weight.dtype))
            out[row, : layout.options] = (keys @ query).float() * LATENT ** -0.5
        return out

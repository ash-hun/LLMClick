"""Supervised fine-tuning on conversations: cross-entropy on every assistant turn, tool calls included."""

import math
from typing import Any, ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from modeling.tuning.training_method import Row, chunks
from modeling.llm.models.llm_backbone import LLMBackbone, Message
from modeling.llm.methods.llm_method import LLMMethod
from core.progress_reporter import Progress


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    arguments: dict[str, Any] | str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    function: ToolFunction


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _calls_from_the_assistant(self) -> "Turn":
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("only an assistant turn makes tool_calls")
        return self


class Conversation(BaseModel):
    """An SFT row: the turns, and the tools the model may call (JSON schemas in the OpenAI shape the chat templates read)."""
    messages: list[Turn] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _well_formed(self) -> "Conversation":
        if self.messages[-1].role != "assistant":
            raise ValueError("the conversation must end with an assistant turn")
        calls = False
        for index, turn in enumerate(self.messages):
            calls = calls or turn.tool_calls is not None
            previous = self.messages[index - 1] if index else None
            if turn.role == "tool" and (previous is None or not (previous.tool_calls or previous.role == "tool")):
                raise ValueError(f"message {index} is a tool result without a tool call before it")
        if calls and not self.tools:
            raise ValueError("a conversation with tool_calls needs `tools`, the functions the model may call")
        return self


class SFT(LLMMethod):
    """Rows: {"messages": [{"role": "system"|"user"|"assistant"|"tool", "content": str, "tool_calls"?: [...]}, ...],
    "tools"?: [...]} ending with an assistant turn. Tool results are context, never targets."""
    version: ClassVar[int] = 2  # 2: earlier assistant turns are graded under templates that mark only the final turn (Qwen3)

    def messages(self, row: Row) -> list[Message]:
        messages: list[Message] = row["messages"]
        return messages

    def tools(self, row: Row) -> list[dict[str, Any]] | None:
        tools: list[dict[str, Any]] | None = row.get("tools")
        return tools

    def check(self, row: Row) -> None:
        try:
            Conversation.model_validate({"messages": row["messages"], "tools": row.get("tools")})
        except ValidationError as error:
            raise ValueError("; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:3])) from None

    def lengths(self, backbone: LLMBackbone, row: Row) -> list[int]:
        return [len(backbone.render(self.messages(row), False, self.tools(row)))]

    def totals(self, backbone: LLMBackbone, rows: list[Row]) -> tuple[torch.Tensor, torch.Tensor]:
        scores, counts = self.log_probabilities(backbone, [encoded for row in rows
                                                           for encoded in self.encode_turns(backbone, self.messages(row), self.tools(row))])
        return -scores.sum(), counts.sum()

    def loss(self, backbone: LLMBackbone, rows: list[Row]) -> torch.Tensor:
        negative, count = self.totals(backbone, rows)
        return negative / count.clamp(min=1)

    def tool_turns(self, rows: list[Row]) -> list[tuple[Row, int]]:
        """Every assistant turn that calls a tool, as (row, turn index)."""
        return [(row, index) for row in rows for index, message in enumerate(self.messages(row))
                if message["role"] == "assistant" and message.get("tool_calls")]

    @torch.no_grad()
    def tool_call_exact(self, backbone: LLMBackbone, turns: list[tuple[Row, int]], batch_size: int, progress: Progress) -> float:
        """The share of tool-calling turns the model writes exactly, greedily, given the conversation before them: the
        same tokens as the turn rendered by the model's own template up to its end, so it holds for any call format."""
        exact, batches = 0, [turns[start:start + batch_size] for start in range(0, len(turns), batch_size)]
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "tool calls")
            prompts, references, longest = [], [], 0
            for row, turn in batch:
                ids, targets = self.encode(backbone, self.messages(row)[:turn + 1], last_only=True, tools=self.tools(row))
                graded = [token for token, flag in zip(ids, targets, strict=True) if flag]
                end = backbone.tokenizer.eos_token_id
                prompts.append(ids[:targets.index(True)])
                references.append(graded[:graded.index(end)] if end in graded else graded)  # generation stops before it
                longest = max(longest, len(graded))
            outputs = backbone.generate(prompts, longest + 4)
            exact += sum(output[0] == reference for output, reference in zip(outputs, references, strict=True))
        return exact / len(turns)

    @torch.no_grad()
    def evaluate(self, backbone: LLMBackbone, rows: list[Row], batch_size: int, progress: Progress) -> dict[str, float]:
        """Loss and perplexity on the assistant turns; `tool_call_exact` as well when the rows call tools."""
        negative, count, batches = 0.0, 0.0, chunks(rows, batch_size)
        for index, batch in enumerate(batches):
            progress.update(index, len(batches), "loss")
            batch_negative, batch_count = self.totals(backbone, batch)
            negative, count = negative + float(batch_negative), count + float(batch_count)
        progress.update(len(batches), len(batches))
        loss = negative / max(count, 1.0)
        metrics = {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "tokens": count}
        turns = self.tool_turns(rows)
        if turns:
            metrics["tool_call_exact"] = self.tool_call_exact(backbone, turns, batch_size, progress)
        return metrics

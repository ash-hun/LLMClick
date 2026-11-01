"""Reading tool calls out of generated text, in the two formats the Qwen chat templates teach: JSON and XML-style."""

import re
import json
from typing import Any
from dataclasses import dataclass

BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Reply:
    """A generated turn: the text outside the calls, the calls, and whether a call block could not be read."""
    content: str
    calls: list[ToolCall]
    malformed: bool


def call_of(block: str) -> ToolCall | None:
    """One `<tool_call>` block: `{"name": ..., "arguments": {...}}` (Qwen3) or `<function=...><parameter=...>` (Qwen3.5)."""
    block = block.strip()
    if (function := FUNCTION.search(block)) is not None:
        return ToolCall(function.group(1), {name: value for name, value in PARAMETER.findall(function.group(2))})
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str):
        return None
    arguments = parsed.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return ToolCall(parsed["name"], arguments) if isinstance(arguments, dict) else None


def parsed(text: str) -> Reply:
    """The turn as text and calls. Generated text is untrusted: a block that is not a well-formed call marks the turn
    malformed instead of raising, and an unclosed `<tool_call>` counts as malformed too."""
    blocks = BLOCK.findall(text)
    calls = [call for block in blocks if (call := call_of(block)) is not None]
    content = BLOCK.sub("", text).strip()
    unclosed = "<tool_call>" in content
    return Reply(content.replace("<tool_call>", "").strip() if unclosed else content, calls,
                 malformed=len(calls) != len(blocks) or unclosed)

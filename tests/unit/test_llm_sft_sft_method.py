"""`llm_sft` modeling/llm/methods/sft_method.py: the graded tokens (assistant turns, tool calls), the row check, tool_call_exact."""

import pytest

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.tuning.tuning_config import TrainingConfig
from modeling.llm.methods.sft_method import SFT
from core.progress_reporter import Progress


def test_targets_are_exactly_the_assistant_turns(backbone: TransformerBackbone) -> None:
    messages = [{"role": "user", "content": "What is 2 plus 3 ?"}, {"role": "assistant", "content": "5"},
                {"role": "user", "content": "What is 4 plus 4 ?"}, {"role": "assistant", "content": "8"}]
    method = SFT(SFT.Config(), TrainingConfig())
    ids, targets = method.encode(backbone, messages)
    assert backbone.tokenizer.decode([i for i, flag in zip(ids, targets, strict=True) if flag]) == "5 <|im_end|> 8 <|im_end|>"
    ids, targets = method.encode(backbone, messages, last_only=True)
    assert backbone.tokenizer.decode([i for i, flag in zip(ids, targets, strict=True) if flag]) == "8 <|im_end|>"


TOOL_TEMPLATE = ("{% if tools %}<|im_start|>system\n{% for t in tools %}{{ t['function']['name'] }} {% endfor %}<|im_end|>\n{% endif %}"
                 "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if m['role'] == 'assistant' and loop.last %}<think> </think> {% endif %}"
                 "{{ m['content'] }}{% if 'tool_calls' in m %}{% for c in m['tool_calls'] %}<|box_start|> {{ c['function']['name'] }} <|box_end|>"
                 "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n<think> </think> {% endif %}")
TOOLS = [{"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {}}}}]
CALLING = [{"role": "user", "content": "What is 2 plus 3 ?"},
           {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "add", "arguments": {}}}]},
           {"role": "tool", "content": "5"}, {"role": "assistant", "content": "5"}]


def graded(backbone: TransformerBackbone, messages: list, tools: list | None = None) -> list[str]:
    sequences = SFT(SFT.Config(), TrainingConfig()).encode_turns(backbone, messages, tools)
    return [str(backbone.tokenizer.decode([i for i, flag in zip(ids, targets, strict=True) if flag])) for ids, targets in sequences]


def test_turns_are_graded_once_each_and_as_written(backbone: TransformerBackbone) -> None:
    messages = [{"role": "user", "content": "What is 2 plus 3 ?"}, {"role": "assistant", "content": "5"},
                {"role": "user", "content": "What is 4 plus 4 ?"}, {"role": "assistant", "content": "8"}]
    assert graded(backbone, messages) == ["5 <|im_end|> 8 <|im_end|>"]
    backbone.tokenizer.chat_template = TOOL_TEMPLATE  # marks the final turn only, as Qwen3's empty thinking block does
    backbone._tokens.clear()
    assert graded(backbone, messages) == ["8 <|im_end|>", "5 <|im_end|>"]
    sequences = SFT(SFT.Config(), TrainingConfig()).encode_turns(backbone, messages)
    assert backbone.tokenizer.decode(sequences[1][0]).endswith("<think> </think> 5 <|im_end|>")


def test_tool_calls_are_targets_and_tool_results_are_not(backbone: TransformerBackbone) -> None:
    backbone.tokenizer.chat_template = TOOL_TEMPLATE
    assert graded(backbone, CALLING, TOOLS) == ["5 <|im_end|>", "<|box_start|> <unk> <|box_end|> <|im_end|>"]


def test_tool_conversations_are_checked() -> None:
    method = SFT(SFT.Config(), TrainingConfig())
    method.check({"messages": CALLING, "tools": TOOLS})
    with pytest.raises(ValueError, match="needs `tools`"):
        method.check({"messages": CALLING})
    with pytest.raises(ValueError, match="without a tool call before it"):
        method.check({"messages": [CALLING[0], CALLING[2], CALLING[3]], "tools": TOOLS})
    with pytest.raises(ValueError, match="only an assistant turn"):
        method.check({"messages": [{**CALLING[0], "tool_calls": CALLING[1]["tool_calls"]}, CALLING[3]], "tools": TOOLS})
    with pytest.raises(ValueError, match="end with an assistant turn"):
        method.check({"messages": CALLING[:3], "tools": TOOLS})


def test_tool_call_exact_compares_the_written_call_with_the_rendered_one(backbone: TransformerBackbone,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    backbone.tokenizer.chat_template = TOOL_TEMPLATE
    method = SFT(SFT.Config(), TrainingConfig())
    row = {"messages": CALLING, "tools": TOOLS}
    ids, targets = method.encode(backbone, CALLING[:2], last_only=True, tools=TOOLS)
    reference = [token for token, flag in zip(ids, targets, strict=True) if flag][:-1]  # without the end-of-turn token
    turns = method.tool_turns([row, {"messages": CALLING[:1] + CALLING[3:]}])
    assert turns == [(row, 1)]
    for written, expected in ((reference, 1.0), (reference[1:], 0.0)):
        monkeypatch.setattr(backbone, "generate", lambda prompts, *a, written=written, **k: [[written] for _ in prompts])
        assert method.tool_call_exact(backbone, turns, 4, Progress()) == expected


THINK_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{% if 'reasoning_content' in m and loop.last %}<think> "
                  "{{ m['reasoning_content'] }} </think> {% endif %}{{ m['content'] }}<|im_end|>\n{% endfor %}"
                  "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def test_reasoning_and_answer_losses_split_the_assistant_tokens_at_the_thinking_block(backbone: TransformerBackbone) -> None:
    backbone.tokenizer.chat_template = THINK_TEMPLATE
    method = SFT(SFT.Config(), TrainingConfig())
    rows = [{"messages": [{"role": "user", "content": f"What is {a} plus 3 ?"},
                          {"role": "assistant", "content": str(a + 3), "reasoning_content": f"{a} and 3"}]} for a in (2, 4)]
    metrics = method.evaluate(backbone, rows, 4, Progress())
    reasoning, answer = 5, 2  # "<think> 2 and 3 </think>", then "5 <|im_end|>"
    assert metrics["tokens"] == 2 * (reasoning + answer)
    expected = (metrics["reasoning_loss"] * reasoning + metrics["answer_loss"] * answer) / (reasoning + answer)
    assert metrics["loss"] == pytest.approx(expected, rel=1e-5) and metrics["reasoning_loss"] != metrics["answer_loss"]
    assert "reasoning_loss" not in method.evaluate(backbone, [{"messages": rows[0]["messages"][:1] + [{"role": "assistant", "content": "5"}]}], 4, Progress())
    with pytest.raises(ValueError, match="only an assistant turn"):
        method.check({"messages": [{"role": "user", "content": "q", "reasoning_content": "r"}, {"role": "assistant", "content": "a"}]})

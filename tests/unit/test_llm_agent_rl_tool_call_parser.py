"""`llm_agent_rl` modeling/llm/agent/tool_call_parser.py: both Qwen call formats, and untrusted text that is not a call."""

from modeling.llm.agent.tool_call_parser import ToolCall, parsed


def test_json_and_xml_calls_are_read() -> None:
    qwen3 = parsed('Let me compute.\n<tool_call>\n{"name": "calculate", "arguments": {"expression": "2*3"}}\n</tool_call>')
    assert qwen3.calls == [ToolCall("calculate", {"expression": "2*3"})] and qwen3.content == "Let me compute." and not qwen3.malformed
    qwen35 = parsed("<tool_call>\n<function=calculate>\n<parameter=expression>\n2*3\n</parameter>\n</function>\n</tool_call>")
    assert qwen35.calls == [ToolCall("calculate", {"expression": "2*3"})]
    assert parsed('<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>').calls == [ToolCall("f", {"a": 1})]


def test_a_final_answer_has_no_calls_and_broken_calls_are_malformed() -> None:
    assert parsed("The answer is 6.") == parsed("The answer is 6.") and parsed("The answer is 6.").calls == []
    assert not parsed("The answer is 6.").malformed
    for text in ('<tool_call>{"name": 3}</tool_call>', "<tool_call>not json</tool_call>", '<tool_call>{"name": "f"'):
        assert parsed(text).malformed

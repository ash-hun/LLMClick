"""`data_synthetic` data/teacher_client.py: the Ollama, OpenAI-format and Anthropic teachers speak their protocols and price calls."""

import io
from typing import Any

import pytest

from data.teacher_client import Completion


def test_ollama_and_openai_teachers_speak_their_protocols(monkeypatch: pytest.MonkeyPatch) -> None:
    from data import teacher_client

    sent: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def fake_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        sent.append((url, body, headers))
        if "/api/chat" in url:
            return {"message": {"content": "ollama says hi"}, "prompt_eval_count": 7, "eval_count": 3}
        return {"choices": [{"message": {"content": "openai says hi"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    monkeypatch.setattr(teacher_client, "post_json", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ollama = teacher_client.Ollama({"model": "qwen3.5:0.8b", "keep_alive": "5m"})
    done = ollama.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done == Completion("ollama says hi", 7, 3, 0.0)
    assert sent[0][0] == "http://localhost:11434/api/chat" and sent[0][1]["options"] == {"temperature": 0.5, "num_predict": 10, "seed": 1}
    assert sent[0][1]["think"] is False and sent[0][1]["keep_alive"] == "5m"
    openai = teacher_client.OpenAIFormat({"model": "gpt-x", "base_url": "http://vllm:8000/v1/", "price": [1, 2]})
    done = openai.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done.text == "openai says hi" and done.cost_usd == pytest.approx((7 * 1 + 3 * 2) / 1_000_000)
    assert sent[1][0] == "http://vllm:8000/v1/chat/completions" and sent[1][2]["authorization"] == "Bearer k"
    assert openai.identity() == {"name": "openai", "model": "gpt-x"}  # keys and prices are not identity
    assert sent[1][1]["max_tokens"] == 10 and sent[1][1]["temperature"] == 0.5
    reasoning = teacher_client.OpenAIFormat({"model": "gpt-r", "reasoning_effort": "low"})
    reasoning.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert sent[-1][1]["max_completion_tokens"] == 10 and sent[-1][1]["reasoning_effort"] == "low"
    assert "temperature" not in sent[-1][1] and "max_tokens" not in sent[-1][1]
    assert reasoning.identity() == {"name": "openai", "model": "gpt-r", "reasoning_effort": "low"}
    teacher_client.Ollama({"model": "m", "think": True}).complete([{"role": "user", "content": "hi"}], max_tokens=1, temperature=0, seed=0)
    assert sent[-1][1]["think"] is True and teacher_client.Ollama({"model": "m"}).identity() == {"name": "ollama", "model": "m", "think": False}


def test_teachers_return_the_reasoning_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    from data import teacher_client

    def fake_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        if "/api/chat" in url:
            return {"message": {"content": "4", "thinking": "2 and 2"}, "prompt_eval_count": 1, "eval_count": 1}
        return {"choices": [{"message": {"content": "4", "reasoning_content": "two twos"}}], "usage": {}}

    monkeypatch.setattr(teacher_client, "post_json", fake_post)
    ask = {"max_tokens": 9, "temperature": 0.0, "seed": 0}
    ollama, openai = teacher_client.Ollama({"model": "m", "think": True}), teacher_client.OpenAIFormat({"model": "m"})
    assert ollama.complete([], **ask, reasoning=True).reasoning == "2 and 2" and ollama.complete([], **ask).reasoning == ""
    assert openai.complete([], **ask, reasoning=True).reasoning == "two twos" and openai.complete([], **ask).reasoning == ""
    with pytest.raises(ValueError, match="summarized thinking"):
        teacher_client.Claude({}).complete([], **ask, reasoning=True)


def test_the_local_teacher_splits_its_thinking_block(tiny_model: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data import teacher_client

    local = teacher_client.Local({"name": str(tiny_model), "architecture": "transformer", "device": "cpu"})
    backbone = local.load()
    monkeypatch.setattr(backbone, "generate", lambda *a, **k: [[backbone.token_ids("<think> 2 and 2 </think> 4")]])
    split = local.complete([{"role": "user", "content": "hi"}], max_tokens=9, temperature=0.0, seed=0, reasoning=True)
    assert (split.reasoning, split.text) == ("2 and 2", "4")
    assert local.complete([{"role": "user", "content": "hi"}], max_tokens=9, temperature=0.0, seed=0).reasoning == ""


def test_the_anthropic_teacher_uses_the_sdk_and_prices_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from data import teacher_client

    class Usage:
        input_tokens, output_tokens = 100, 50

    class Block:
        type, text = "text", "claude says hi"

    class Response:
        stop_reason, content, usage = "end_turn", [Block()], Usage()

    class Messages:
        def create(self, **keys: Any) -> Response:
            assert keys["model"] == "claude-opus-5-5" and keys["system"] == "be brief" and keys["messages"][0]["role"] == "user"
            return Response()

    class Client:
        messages = Messages()

    teacher = teacher_client.Claude({})
    teacher.client = Client()
    done = teacher.complete([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done.text == "claude says hi" and done.cost_usd == pytest.approx(100 * 4 / 1_000_000 + 50 * 20 / 1_000_000)
    monkeypatch.setattr(io, "BytesIO", io.BytesIO)  # keep the import used

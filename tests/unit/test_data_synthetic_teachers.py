"""`data_synthetic` data/teachers.py: the Ollama, OpenAI-format and Anthropic teachers speak their protocols and price calls."""

import io
from typing import Any

import pytest

from data.teachers import Completion


def test_ollama_and_openai_teachers_speak_their_protocols(monkeypatch: pytest.MonkeyPatch) -> None:
    from data import teachers

    sent: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def fake_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
        sent.append((url, body, headers))
        if "/api/chat" in url:
            return {"message": {"content": "ollama says hi"}, "prompt_eval_count": 7, "eval_count": 3}
        return {"choices": [{"message": {"content": "openai says hi"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    monkeypatch.setattr(teachers, "post_json", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    ollama = teachers.Ollama({"model": "qwen3.5:0.8b", "keep_alive": "5m"})
    done = ollama.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done == Completion("ollama says hi", 7, 3, 0.0)
    assert sent[0][0] == "http://localhost:11434/api/chat" and sent[0][1]["options"] == {"temperature": 0.5, "num_predict": 10, "seed": 1}
    assert sent[0][1]["think"] is False and sent[0][1]["keep_alive"] == "5m"
    openai = teachers.OpenAIFormat({"model": "gpt-x", "base_url": "http://vllm:8000/v1/", "price": [1, 2]})
    done = openai.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done.text == "openai says hi" and done.cost_usd == pytest.approx((7 * 1 + 3 * 2) / 1_000_000)
    assert sent[1][0] == "http://vllm:8000/v1/chat/completions" and sent[1][2]["authorization"] == "Bearer k"
    assert openai.identity() == {"name": "openai", "model": "gpt-x"}  # keys and prices are not identity
    assert sent[1][1]["max_tokens"] == 10 and sent[1][1]["temperature"] == 0.5
    reasoning = teachers.OpenAIFormat({"model": "gpt-r", "reasoning_effort": "low"})
    reasoning.complete([{"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert sent[-1][1]["max_completion_tokens"] == 10 and sent[-1][1]["reasoning_effort"] == "low"
    assert "temperature" not in sent[-1][1] and "max_tokens" not in sent[-1][1]
    assert reasoning.identity() == {"name": "openai", "model": "gpt-r", "reasoning_effort": "low"}
    teachers.Ollama({"model": "m", "think": True}).complete([{"role": "user", "content": "hi"}], max_tokens=1, temperature=0, seed=0)
    assert sent[-1][1]["think"] is True and teachers.Ollama({"model": "m"}).identity() == {"name": "ollama", "model": "m", "think": False}


def test_the_anthropic_teacher_uses_the_sdk_and_prices_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from data import teachers

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

    teacher = teachers.Claude({})
    teacher.client = Client()
    done = teacher.complete([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], max_tokens=10, temperature=0.5, seed=1)
    assert done.text == "claude says hi" and done.cost_usd == pytest.approx(100 * 4 / 1_000_000 + 50 * 20 / 1_000_000)
    monkeypatch.setattr(io, "BytesIO", io.BytesIO)  # keep the import used

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from jeff.teacher import MalformedOutput, Teacher, TeacherError

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def reply(content: str | None, finish: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": {"content": content}}],
                                     "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def run(handler, tmp_path: Path, **call):
    calls: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(len(calls))

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
            teacher = Teacher(client, url="http://t/v1", model="m", cache=tmp_path / "cache", log=tmp_path / "log.jsonl",
                              retry_delays=(0.0, 0.0))
            return await teacher.complete(system="s", user="u", schema=SCHEMA, temperature=0.0, max_tokens=10, **call)

    return asyncio.run(go()), calls


def test_returns_parsed_json_and_sends_schema(tmp_path: Path) -> None:
    result, calls = run(lambda n: reply('{"x": 1}'), tmp_path)
    assert result == {"x": 1}
    body = json.loads(calls[0].content)
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_cache_hit_skips_http(tmp_path: Path) -> None:
    run(lambda n: reply('{"x": 1}'), tmp_path)
    result, calls = run(lambda n: pytest.fail("HTTP must not be called on a cache hit"), tmp_path)
    assert result == {"x": 1} and calls == []


def test_truncated_reply_is_malformed(tmp_path: Path) -> None:
    with pytest.raises(MalformedOutput, match="length"):
        run(lambda n: reply('{"x": ', finish="length"), tmp_path)


def test_invalid_json_is_malformed(tmp_path: Path) -> None:
    with pytest.raises(MalformedOutput):
        run(lambda n: reply("not json"), tmp_path)


def test_retries_server_errors_then_succeeds(tmp_path: Path) -> None:
    result, calls = run(lambda n: httpx.Response(503) if n < 3 else reply('{"x": 2}'), tmp_path)
    assert result == {"x": 2} and len(calls) == 3
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2


def test_gives_up_after_three_tries(tmp_path: Path) -> None:
    with pytest.raises(TeacherError, match="3 tries"):
        run(lambda n: httpx.Response(502), tmp_path)


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        run(lambda n: httpx.Response(400, json={"error": "bad schema"}), tmp_path)


def test_never_more_than_max_in_flight_requests(tmp_path: Path) -> None:
    active = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return reply('{"x": 1}')

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            teacher = Teacher(client, url="http://t/v1", model="m", cache=tmp_path / "cache", log=tmp_path / "log.jsonl",
                              max_in_flight=3)
            await asyncio.gather(*(teacher.complete(system="s", user=f"u{i}", schema=SCHEMA, temperature=0.0, max_tokens=10)
                                   for i in range(20)))

    asyncio.run(go())
    assert peak == 3

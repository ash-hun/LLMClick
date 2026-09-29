"""Client for the local teacher model (vLLM, OpenAI-compatible) with a disk cache and bounded retries."""

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from jeff.types import JSONValue

TEACHER_MODEL = "qwen3.8-flash-next"


def teacher_url() -> str:
    """The teacher's OpenAI-compatible endpoint, from JEFF_TEACHER_URL (for example http://my-server:8888/v1)."""
    url = os.environ.get("JEFF_TEACHER_URL")
    if not url:
        raise RuntimeError("Set JEFF_TEACHER_URL to the teacher model's OpenAI-compatible endpoint, e.g. http://my-server:8888/v1")
    return url
TRIES = 3


class MalformedOutput(ValueError):
    """The teacher replied, but the reply is truncated or is not valid JSON."""


class TeacherError(RuntimeError):
    """The teacher could not be reached after every allowed try."""


class Retryable(Exception):
    pass


class Teacher:
    def __init__(self, client: httpx.AsyncClient, *, url: str, model: str, cache: Path, log: Path,
                 retry_delays: tuple[float, float] = (5.0, 20.0), max_in_flight: int = 8) -> None:
        self.client, self.url, self.model, self.cache, self.log = client, url.rstrip("/"), model, cache, log
        self.retry_delays = retry_delays
        # The teacher serves 8 requests at once and is shared; never send more than this, whatever the caller does.
        self.gate = asyncio.Semaphore(max_in_flight)

    async def complete(self, *, system: str, user: str, schema: dict[str, object], temperature: float,
                       max_tokens: int, seed: int | None = None) -> dict[str, JSONValue]:
        """`seed` makes repeated sampling of the same request give different, reproducible answers (and cache keys)."""
        payload = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "response_format": {"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}},
                   "chat_template_kwargs": {"enable_thinking": False}}
        if seed is not None:
            payload["seed"] = seed
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache / key[:2] / f"{key}.json"
        if path.exists():
            body = json.loads(path.read_text())
        else:
            async with self.gate:
                body = await self.post(payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(body))
            temporary.replace(path)
        return parse(body)

    async def post(self, payload: dict[str, object]) -> dict[str, JSONValue]:
        for attempt in range(TRIES):
            try:
                response = await self.client.post(f"{self.url}/chat/completions", json=payload, timeout=600)
                if response.status_code == 429 or response.status_code >= 500:
                    raise Retryable(f"HTTP {response.status_code}")
                response.raise_for_status()
                result: dict[str, JSONValue] = response.json()
                return result
            except (httpx.TransportError, Retryable) as error:
                if attempt == TRIES - 1:
                    raise TeacherError(f"Teacher unreachable after {TRIES} tries: {error!r}") from error
                with self.log.open("a") as stream:
                    stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": "teacher_retry",
                                             "attempt": attempt + 1, "error": repr(error)}) + "\n")
                await asyncio.sleep(self.retry_delays[attempt])
        raise AssertionError("unreachable")


def parse(body: dict[str, JSONValue]) -> dict[str, JSONValue]:
    choice = body["choices"][0]  # type: ignore[index]
    if choice["finish_reason"] != "stop":  # type: ignore[index,call-overload]
        raise MalformedOutput(f"Reply ended with finish_reason={choice['finish_reason']}")  # type: ignore[index,call-overload]
    content = choice["message"]["content"]  # type: ignore[index,call-overload]
    try:
        value = json.loads(content)  # type: ignore[arg-type]
    except (TypeError, json.JSONDecodeError) as error:
        raise MalformedOutput(f"Reply is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise MalformedOutput("Reply JSON is not an object")
    return value

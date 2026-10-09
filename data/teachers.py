"""The models that write rows: four providers behind one `complete`, each reporting tokens and cost."""

import os
import json
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.request import Request, urlopen

from core.registry import Registry

TEACHERS = Registry("teacher")
Message = dict[str, str]
MILLION = 1_000_000
# USD per million tokens (input, output) for the models a config is likely to name; others need `price` in the params.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
}


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class Teacher:
    """One provider. `identity` is what makes its output differ (it joins the stage fingerprints); `complete`
    is the only call that leaves the process."""
    name: ClassVar[str]

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.model = str(params.get("model", ""))
        price = params.get("price") or PRICES.get(self.model) or (0.0, 0.0)
        self.price = (float(price[0]), float(price[1]))

    def identity(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, **{k: v for k, v in self.params.items() if k not in {"api_key_env", "price", "base_url"}}}

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.price[0] / MILLION + output_tokens * self.price[1] / MILLION

    def complete(self, messages: list[Message], *, max_tokens: int, temperature: float, seed: int) -> Completion:
        raise NotImplementedError


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers}, method="POST")
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - the host comes from the config
        payload: dict[str, Any] = json.loads(response.read())
    return payload


@TEACHERS.register("ollama")
class Ollama(Teacher):
    """A local Ollama server through its native chat API (`base_url`, default http://localhost:11434)."""
    name = "ollama"

    def complete(self, messages: list[Message], *, max_tokens: int, temperature: float, seed: int) -> Completion:
        base = str(self.params.get("base_url", "http://localhost:11434")).rstrip("/")
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
                **({"keep_alive": self.params["keep_alive"]} if "keep_alive" in self.params else {})}
        payload = post_json(f"{base}/api/chat", body, {}, float(self.params.get("timeout", 600)))
        prompt, generated = int(payload.get("prompt_eval_count", 0)), int(payload.get("eval_count", 0))
        return Completion(str(payload["message"]["content"]), prompt, generated, self.cost(prompt, generated))


@TEACHERS.register("openai")
class OpenAIFormat(Teacher):
    """Any server with the OpenAI chat-completions shape: OpenAI, vLLM, LM Studio, a gateway. The key comes from
    the environment variable `api_key_env` names (default OPENAI_API_KEY)."""
    name = "openai"

    def complete(self, messages: list[Message], *, max_tokens: int, temperature: float, seed: int) -> Completion:
        base = str(self.params.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        key = os.environ.get(str(self.params.get("api_key_env", "OPENAI_API_KEY")), "")
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        payload = post_json(f"{base}/chat/completions", body, {"authorization": f"Bearer {key}"} if key else {},
                            float(self.params.get("timeout", 600)))
        usage = payload.get("usage") or {}
        prompt, generated = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        return Completion(str(payload["choices"][0]["message"]["content"]), prompt, generated, self.cost(prompt, generated))


@TEACHERS.register("anthropic")
class Claude(Teacher):
    """The Claude API through the official SDK; the key comes from ANTHROPIC_API_KEY. Thinking stays at the model's
    default (adaptive on current models); `max_tokens` bounds the visible answer."""
    name = "anthropic"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__({"model": "claude-opus-5-5", **params})
        self.client: Any = None

    def complete(self, messages: list[Message], *, max_tokens: int, temperature: float, seed: int) -> Completion:
        if self.client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError("the anthropic teacher needs the SDK: `uv sync --extra data`") from None
            self.client = anthropic.Anthropic()
        system = [message["content"] for message in messages if message["role"] == "system"]
        turns = [message for message in messages if message["role"] != "system"]
        response = self.client.messages.create(model=self.model, max_tokens=max_tokens, messages=turns,
                                               **({"system": "\n\n".join(system)} if system else {}))
        if response.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        prompt, generated = int(response.usage.input_tokens), int(response.usage.output_tokens)
        return Completion(text, prompt, generated, self.cost(prompt, generated))


@TEACHERS.register("local")
class Local(Teacher):
    """A checkpoint this repository trained, or a Hub model: `experiment` + `checkpoint`, or `name` + `revision`
    + `architecture`, as the evaluation recipes name models. Free, and seeded, so the same request gives the same
    text on the same hardware."""
    name = "local"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self.backbone: Any = None

    def identity(self) -> dict[str, Any]:
        from evaluation.config import SourceModel
        from evaluation.source import SourceExperiment
        model = SourceModel(**{k: v for k, v in self.params.items() if k in SourceModel.model_fields})
        return {"name": self.name, "model": SourceExperiment(model).identity(), "device": self.params.get("device")}

    def load(self) -> Any:
        from core.utils.device import resolve_device
        from evaluation.config import SourceModel
        from evaluation.source import SourceExperiment
        from modeling.llm.models.base import LLMBackbone
        if self.backbone is None:
            source = SourceExperiment(SourceModel(**{k: v for k, v in self.params.items() if k in SourceModel.model_fields}))
            backbone = source.hub_backbone() if source.hub else source.build()[0]
            if not isinstance(backbone, LLMBackbone):
                raise ValueError("the local teacher needs a language model")
            backbone.load(resolve_device(self.params.get("device")), source.checkpoint())
            self.backbone = backbone
        return self.backbone

    def complete(self, messages: list[Message], *, max_tokens: int, temperature: float, seed: int) -> Completion:
        from core.utils import device
        backbone = self.load()
        device.seed(backbone.device, seed)
        prompt = backbone.render(messages, True)
        tokens = backbone.generate([prompt], max_tokens, 1, temperature)[0][0]
        return Completion(backbone.text(tokens), len(prompt), len(tokens), 0.0)

    def release(self) -> None:
        if self.backbone is not None:
            self.backbone.release()
            self.backbone = None


def teacher_from(spec: Any) -> Teacher:
    """A `Keyed` config entry (`name` plus parameters) to its teacher."""
    cls: type[Teacher] = TEACHERS.get(spec.name)
    return cls(spec.params)

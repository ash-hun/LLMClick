"""Environments `llm_agent_rl` rolls out in: the tools a policy may call, and what calling them returns."""

import ast
import json
import operator
from pathlib import Path
from typing import Any, ClassVar
from collections.abc import Callable

from core.recipe_registry import Registry

ENVIRONMENTS = Registry("environment")
RESULT_CHARS = 1000  # a tool result is cut here: the next prompt must still fit max_length
OPERATORS: dict[type, Callable[..., Any]] = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                                             ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                                             ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
                                             ast.UAdd: operator.pos}
MAX_EXPRESSION, MAX_EXPONENT, MAX_MAGNITUDE = 200, 64, 10 ** 30


def function(name: str, description: str, parameters: dict[str, str]) -> dict[str, Any]:
    """A tool schema in the OpenAI shape the chat templates read; every parameter is a required string."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {key: {"type": "string", "description": text} for key, text in parameters.items()},
        "required": list(parameters)}}}


class Environment:
    """One rollout's tools. `call` answers with text; a failed call answers with the error, as a real tool would,
    so the policy can read it and try again. A subclass sets `name`, `tools` and its handlers."""
    name: ClassVar[str]

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    @property
    def tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{type(self).__name__} must define tools")

    def handlers(self) -> dict[str, Callable[..., str]]:
        raise NotImplementedError(f"{type(self).__name__} must define handlers")

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        handler = self.handlers().get(name)
        if handler is None:
            return f"error: no tool named {name!r}; the tools are {sorted(self.handlers())}"
        try:
            result = handler(**{key: str(value) for key, value in arguments.items()})
        except TypeError as error:
            return f"error: {error}"
        return result[:RESULT_CHARS]


def evaluated(node: ast.AST) -> Any:
    """Arithmetic only: numbers and the operators in OPERATORS; names, calls and attributes never evaluate."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](evaluated(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = evaluated(node.left), evaluated(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        value = OPERATORS[type(node.op)](left, right)
        if abs(value) > MAX_MAGNITUDE:
            raise ValueError("result too large")
        return value
    raise ValueError("only numbers and + - * / // % ** with parentheses")


@ENVIRONMENTS.register("calculator")
class Calculator(Environment):
    """`calculate(expression)`: arithmetic on numbers, parsed rather than run."""
    name = "calculator"

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [function("calculate", "Evaluate an arithmetic expression such as (12 + 7) * 3",
                         {"expression": "The expression, with numbers and + - * / // % ** ( )"})]

    def handlers(self) -> dict[str, Callable[..., str]]:
        return {"calculate": self.calculate}

    def calculate(self, expression: str) -> str:
        if len(expression) > MAX_EXPRESSION:
            return f"error: the expression is longer than {MAX_EXPRESSION} characters"
        try:
            value = evaluated(ast.parse(expression.replace(",", ""), mode="eval").body)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
            return f"error: {error}"
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


@ENVIRONMENTS.register("lookup")
class Lookup(Environment):
    """`search(query)` over the documents of `path` (a jsonl of {"title", "text"}): the `top` documents sharing the
    most words with the query."""
    name = "lookup"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        path = params.get("path")
        if not isinstance(path, str):
            raise ValueError("the lookup environment needs `path`, a jsonl of {\"title\", \"text\"} documents")
        self.documents = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.top = int(params.get("top", 2))

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [function("search", "Search the documents", {"query": "Words to look for"})]

    def handlers(self) -> dict[str, Callable[..., str]]:
        return {"search": self.search}

    def search(self, query: str) -> str:
        words = set(query.lower().split())
        ranked = sorted(self.documents, key=lambda doc: -len(words & set(f"{doc['title']} {doc['text']}".lower().split())))
        return json.dumps([{"title": doc["title"], "text": doc["text"]} for doc in ranked[: self.top]], ensure_ascii=False)

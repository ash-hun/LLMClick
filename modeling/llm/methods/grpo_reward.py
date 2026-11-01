"""Reward registry that `llm_grpo` and `data_synthetic` verify score a completion with: (completion, row, params) -> float."""

import re
from typing import Any

from modeling.tuning.training_method import Row
from core.recipe_registry import Registry

REWARDS = Registry("reward")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
MODES = {"search": re.search, "full": re.fullmatch}


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


@REWARDS.register("exact_match")
def exact_match(completion: str, row: Row, params: dict[str, Any]) -> float:
    """1 when the completion is the row's `answer` (case and spacing ignored)."""
    return float(normalized(completion) == normalized(str(row["answer"])))


@REWARDS.register("contains")
def contains(completion: str, row: Row, params: dict[str, Any]) -> float:
    """1 when the row's `answer` appears in the completion."""
    return float(normalized(str(row["answer"])) in normalized(completion))


def last_number(text: str) -> float | None:
    found = NUMBER.findall(text)
    return float(found[-1].replace(",", "")) if found else None


@REWARDS.register("last_number")
def last_number_match(completion: str, row: Row, params: dict[str, Any]) -> float:
    """1 when the last number in the completion equals the last number in the row's `answer`; this reads worked
    solutions that end with their result, such as GSM8K's "... #### 18"."""
    expected = last_number(str(row["answer"]))
    if expected is None:
        raise ValueError("answer holds no number")
    return float(last_number(completion) == expected)


@REWARDS.register("regex")
def regex_match(completion: str, row: Row, params: dict[str, Any]) -> float:
    """1 when the completion matches `pattern`: anywhere (`mode: search`, the default) or as a whole (`mode: full`).
    A format reward, e.g. `<think>.*</think>` before the answer."""
    if "pattern" not in params:
        raise ValueError("the regex reward needs a `pattern`")
    mode = str(params.get("mode", "search"))
    if mode not in MODES:
        raise ValueError(f"regex mode is {mode!r}; one of {sorted(MODES)}")
    return float(MODES[mode](str(params["pattern"]), completion, re.DOTALL) is not None)


@REWARDS.register("length")
def length_limit(completion: str, row: Row, params: dict[str, Any]) -> float:
    """1 up to `max_chars` characters, then falling linearly to 0 at twice that: discourages padding an answer."""
    limit = int(params.get("max_chars", 0))
    if limit < 1:
        raise ValueError("the length reward needs `max_chars` of at least 1")
    return max(0.0, min(1.0, 2.0 - len(completion) / limit))


def parts(params: dict[str, Any]) -> list[dict[str, Any]]:
    """The weighted reward's `parts`, each a registered reward other than `weighted` itself with a numeric weight."""
    entries = params.get("parts")
    if not isinstance(entries, list) or not entries:
        raise ValueError("the weighted reward needs a non-empty `parts` list")
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("weight", 1.0), (int, float)):
            raise ValueError("every weighted part is a mapping with `name` and an optional numeric `weight`")
        if entry.get("name") == "weighted":
            raise ValueError("a weighted reward cannot contain another weighted reward")
        REWARDS.check(str(entry.get("name")))
    return entries


@REWARDS.register("weighted")
def weighted(completion: str, row: Row, params: dict[str, Any]) -> float:
    """Sum of `parts`, each `{name, weight (default 1), ...its own params}`, e.g. correctness plus a small format term."""
    total = 0.0
    for entry in parts(params):
        own = {key: value for key, value in entry.items() if key not in {"name", "weight"}}
        total += float(entry.get("weight", 1.0)) * float(REWARDS.get(entry["name"])(completion, row, own))
    return total


def reward_names(name: str, params: dict[str, Any]) -> list[str]:
    """Every reward a config entry uses, for the registry check: a weighted reward names its parts too."""
    return [name, *(str(entry["name"]) for entry in parts(params))] if name == "weighted" else [name]

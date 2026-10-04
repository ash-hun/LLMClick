"""Read jeff's training.jsonl and evaluations.jsonl back as (step, metrics) points for the tracker."""

import json
from pathlib import Path
from typing import Any

from modeling.tracker import History

FILES = {"training.jsonl": ("loss", "learning_rate", "gradient_norm", "input_tokens"),
         "evaluations.jsonl": ("temperature",)}
NESTED = {"evaluations.jsonl": ("raw", "fitted")}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().split("\n") if line] if path.exists() else []


def flatten(run: Path) -> History:
    points: dict[int, dict[str, float]] = {}
    for name, keys in FILES.items():
        for event in rows(run / name):
            step = int(event["step"])
            metrics = points.setdefault(step, {})
            metrics.update({f"{name[:-6]}/{key}": float(event[key]) for key in keys if key in event})
            for group in NESTED.get(name, ()):
                metrics.update({f"{group}/{key}": float(value) for key, value in (event.get(group) or {}).items()
                                if isinstance(value, (int, float))})
    return sorted(points.items())

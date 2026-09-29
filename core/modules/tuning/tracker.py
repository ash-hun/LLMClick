"""Replay jeff's training.jsonl and evaluations.jsonl into wandb after a run; jeff itself logs only to files."""

import json
import logging
from pathlib import Path
from typing import Any

from core.config.schema import TrackerConfig

logger = logging.getLogger(__name__)
FILES = {"training.jsonl": ("loss", "learning_rate", "gradient_norm", "input_tokens"),
         "evaluations.jsonl": ("temperature",)}
NESTED = {"evaluations.jsonl": ("raw", "fitted")}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().split("\n") if line] if path.exists() else []


def flatten(run: Path) -> list[tuple[int, dict[str, float]]]:
    """(step, metrics) pairs in step order; pure so it can be tested without wandb."""
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


# ponytail: post-hoc replay, not live logging; switch to tailing events.jsonl when someone watches runs in wandb live.
def sync(config: TrackerConfig, run: Path, experiment_key: str, pipeline_config: dict[str, Any]) -> None:
    if not config.enabled:
        return
    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed (uv sync --extra tracker); skipping tracker sync")
        return
    session = wandb.init(project=config.project, entity=config.entity, name=experiment_key, id=experiment_key,
                         resume="allow", tags=config.tags, config=pipeline_config)
    for step, metrics in flatten(run):
        wandb.log(metrics, step=step)
    session.finish()

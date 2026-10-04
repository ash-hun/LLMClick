"""Replay a finished training run's metrics into wandb; the run id is the training directory, so a replay updates it."""

import logging
from typing import Any

from modeling.config import TrackerConfig

logger = logging.getLogger(__name__)
History = list[tuple[int, dict[str, float]]]  # (step, metrics) in step order


class Tracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config

    # ponytail: post-hoc replay, not live logging; log from the progress poll when someone watches runs in wandb live.
    def sync(self, run_id: str, name: str, history: History, settings: dict[str, Any]) -> None:
        if not self.config.enabled:
            return
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed (uv sync --extra tracker); skipping tracker sync")
            return
        session = wandb.init(project=self.config.project, entity=self.config.entity, name=name, id=run_id,
                             resume="allow", tags=self.config.tags, config=settings)
        for step, metrics in history:
            wandb.log(metrics, step=step)
        session.finish()

"""Send a training run's metrics to wandb exactly once per run and target, whenever tracking is switched on."""

import logging
from pathlib import Path
from typing import Any

from core.utils.files import read_json, write_json
from modeling.config import TrackerConfig

logger = logging.getLogger(__name__)
History = list[tuple[int, dict[str, float]]]  # (step, metrics) in step order
MARKER = "tracker.json"


class Tracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config

    @property
    def target(self) -> str:
        return f"{self.config.entity or ''}/{self.config.project}"

    # ponytail: post-hoc replay, not live logging; log from the progress poll when someone watches runs in wandb live.
    def sync(self, run: Path, name: str, history: History, settings: dict[str, Any]) -> bool:
        """Replay `history` into wandb unless this run directory was already sent to this target. The run id is the
        directory name, so one training run is one wandb run however many experiments share it. `tracker` is not
        part of an experiment's identity, so this also covers switching it on after the training was cached."""
        marker = run / MARKER
        synced: list[str] = read_json(marker)["synced"] if marker.exists() else []
        if not self.config.enabled or self.target in synced or not history:
            return False
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed (uv sync --extra tracker); skipping tracker sync")
            return False
        session = wandb.init(project=self.config.project, entity=self.config.entity, name=name, id=run.name,
                             resume="allow", tags=self.config.tags, config=settings)
        for step, metrics in history:
            wandb.log(metrics, step=step)
        session.finish()
        write_json(marker, {"synced": [*synced, self.target]})
        return True

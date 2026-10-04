"""Send a training run's metrics to wandb: live while it trains, or replayed once if tracking is switched on later."""

import logging
from pathlib import Path
from typing import Any

from core.utils.files import read_json, write_json
from modeling.config import TrackerConfig

logger = logging.getLogger(__name__)
History = list[tuple[int, dict[str, float]]]  # (step, metrics) in step order
MARKER = "tracker.json"


class Tracker:
    """One training run directory is one wandb run (the run id is the directory name), however many experiments
    share it. `tracker.json` in that directory lists the targets it was sent to, so nothing is sent twice."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.session: Any = None

    @property
    def target(self) -> str:
        return f"{self.config.entity or ''}/{self.config.project}"

    def synced(self, run: Path) -> list[str]:
        marker = run / MARKER
        targets: list[str] = read_json(marker)["synced"] if marker.exists() else []
        return targets

    def open(self, run: Path, name: str, settings: dict[str, Any]) -> bool:
        """Start (or, after a crash, resume) the wandb run for `run`; False when there is nothing to send to."""
        if not self.config.enabled or self.target in self.synced(run):
            return False
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed (uv sync --extra tracker); training is not tracked")
            return False
        self.session = wandb.init(project=self.config.project, entity=self.config.entity, name=name, id=run.name,
                                  resume="allow", tags=self.config.tags, config=settings)
        return True

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """One training step, as it happens; does nothing without an open run."""
        if self.session is not None:
            self.session.log(metrics, step=step)

    def close(self, run: Path, complete: bool) -> None:
        """End the wandb run. Only a completed training is recorded as sent: an interrupted one is resumed, and
        continues the same wandb run, when the pipeline is run again."""
        if self.session is None:
            return
        self.session.finish()
        self.session = None
        if complete:
            write_json(run / MARKER, {"synced": [*self.synced(run), self.target]})

    def sync(self, run: Path, name: str, history: History, settings: dict[str, Any]) -> bool:
        """Replay a finished run that was never sent to this target: tracking switched on after the training was
        cached, or a second project. `tracker` is not part of an experiment's identity, so this never retrains."""
        if not history or not self.open(run, name, settings):
            return False
        for step, metrics in history:
            self.log(step, metrics)
        self.close(run, complete=True)
        return True

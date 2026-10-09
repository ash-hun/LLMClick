"""Where results live: stage directories named by fingerprint and shared, experiment directories that point at them."""

import os
import logging
import threading
from pathlib import Path
from typing import Any
from collections.abc import Iterator
from contextlib import contextmanager

import yaml

from core.utils.files import read_json, sha256_json, write_json
from core.config.schema import BaseConfig

HASH_LENGTH = 8
STAGE_HASH_LENGTH = 12
STORE = "_stages"
DONE = "stage.json"
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def config_hash(config: BaseConfig) -> str:
    return sha256_json(config.identity())[:HASH_LENGTH]


class Experiment:
    def __init__(self, config: BaseConfig) -> None:
        self.config = config
        self.key = f"{config.name}-{config_hash(config)}"
        self.root = Path(config.output_dir) / self.key
        self.store = Path(config.output_dir) / STORE
        self.manifest_path = self.root / "manifest.json"

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        dumped = yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=True, allow_unicode=True)
        config_file = self.root / "config.yaml"
        if not config_file.exists() or config_file.read_text() != dumped:
            config_file.write_text(dumped)
        if not self.manifest_path.exists():
            write_json(self.manifest_path, {"key": self.key, "recipe": self.config.recipe, "stages": {}})
        return self.root

    def manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = read_json(self.manifest_path)
        return manifest

    def stage_dir(self, stage: str, fingerprint: str) -> Path:
        """The same stage with the same inputs lands in the same directory, whichever experiment asks."""
        return self.store / f"{stage}-{fingerprint[:STAGE_HASH_LENGTH]}"

    def stage_outputs(self, workdir: Path, fingerprint: str) -> dict[str, Any] | None:
        """Outputs recorded in this stage directory if it finished with the same inputs and its files still exist."""
        marker = workdir / DONE
        if not marker.exists():
            return None
        entry = read_json(marker)
        if entry["fingerprint"] != fingerprint:
            return None
        outputs: dict[str, Any] = entry["outputs"]
        inside = [value for value in outputs.values() if isinstance(value, str) and value.startswith(str(workdir))]
        return outputs if all(Path(value).exists() for value in inside) else None

    def mark_done(self, workdir: Path, fingerprint: str, outputs: dict[str, Any]) -> dict[str, Any]:
        """Record the stage as finished and return its outputs as stored, so a first run and a cached run agree."""
        write_json(workdir / DONE, {"fingerprint": fingerprint, "outputs": outputs})
        stored: dict[str, Any] = read_json(workdir / DONE)["outputs"]
        return stored

    def link(self, stage: str, fingerprint: str, workdir: Path, outputs: dict[str, Any]) -> None:
        """Point this experiment at a finished stage: a manifest entry and `<experiment>/<stage>` -> stage directory."""
        manifest = self.manifest()
        entry = {"fingerprint": fingerprint, "directory": str(workdir), "outputs": outputs}
        if manifest["stages"].get(stage) != entry:
            manifest["stages"][stage] = entry
            write_json(self.manifest_path, manifest)
        link = self.root / stage
        target = os.path.relpath(workdir, self.root)
        if link.is_symlink() and os.readlink(link) == target:
            return
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    @contextmanager
    def logging(self) -> Iterator[None]:
        """Copy this run's log records to `<experiment>/pipeline.log` for the duration of one run only. Only records
        from the thread that runs the pipeline are copied: the API runs several jobs at once in one process, each
        on its own worker thread, and the root logger is shared by all of them."""
        handler = logging.FileHandler(self.root / "pipeline.log")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        thread = threading.get_ident()
        handler.addFilter(lambda record: record.thread == thread)
        logging.getLogger().addHandler(handler)
        try:
            yield
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()

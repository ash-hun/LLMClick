"""The experiment directory: named by config hash so the same config always lands in the same place, with a stage manifest."""

import logging
from pathlib import Path
from typing import Any

import yaml

from core.config.schema import PipelineConfig
from core.utils.files import read_json, sha256_json, write_json

logger = logging.getLogger(__name__)
HASH_LENGTH = 8


def load_config(path: str | Path) -> PipelineConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    pipeline = raw.pop("pipeline", {})
    return PipelineConfig(**{**pipeline, **raw})


def config_hash(config: PipelineConfig) -> str:
    return sha256_json(config.model_dump(mode="json"))[:HASH_LENGTH]


class Experiment:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.key = f"{config.name}-{config_hash(config)}"
        self.root = Path(config.output_dir) / self.key
        self.data = self.root / "data"
        self.runs = self.root / "runs"
        self.checkpoints = self.root / "checkpoints"
        self.eval = self.root / "eval"
        self.manifest_path = self.root / "manifest.json"

    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        config_file = self.root / "config.yaml"
        if not config_file.exists():
            config_file.write_text(yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=True, allow_unicode=True))
        if not self.manifest_path.exists():
            write_json(self.manifest_path, {"key": self.key, "stages": {}})
        handler_exists = any(getattr(h, "baseFilename", None) == str(self.root / "pipeline.log")
                             for h in logging.getLogger().handlers)
        if not handler_exists:
            handler = logging.FileHandler(self.root / "pipeline.log")
            handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
            logging.getLogger().addHandler(handler)
        return self.root

    def manifest(self) -> dict[str, Any]:
        return read_json(self.manifest_path) if self.manifest_path.exists() else {"key": self.key, "stages": {}}

    def stage_outputs(self, stage: str, fingerprint: str) -> dict[str, Any] | None:
        """Outputs recorded for this stage if it finished with the same inputs and its files still exist."""
        entry = self.manifest()["stages"].get(stage)
        if not entry or entry["fingerprint"] != fingerprint:
            return None
        outputs: dict[str, Any] = entry["outputs"]
        if all(Path(value).exists() for value in outputs.values() if isinstance(value, str) and value.startswith(str(self.root))):
            return outputs
        return None

    def mark_done(self, stage: str, fingerprint: str, outputs: dict[str, Any]) -> None:
        manifest = self.manifest()
        manifest["stages"][stage] = {"fingerprint": fingerprint, "outputs": outputs}
        write_json(self.manifest_path, manifest)

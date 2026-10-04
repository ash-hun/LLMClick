"""Config sections every modeling recipe shares: experiment tracking and the validation gate."""

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from core.config.schema import BaseConfig
from core.utils.files import sha256_file


class Keyed(BaseModel):
    """A registry entry: `name` picks the implementation, every other key is passed to it as a parameter."""
    model_config = ConfigDict(extra="allow")
    name: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


def keyed_identity(entries: list[Keyed]) -> list[Any]:
    """Registry entries plus the content of any local file they read, so an edited file is a new input."""
    return [[entry.model_dump(mode="json"), sha256_file(Path(path)) if (path := entry.params.get("path"))
             and Path(path).is_file() else None] for entry in entries]


class TrackerConfig(BaseModel):
    enabled: bool = False
    project: str = "llmclick"
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)


class ValidationConfig(BaseModel):
    """Bounds the trained model must meet on held-out data; a metric outside them fails the pipeline."""
    min: dict[str, float] = Field(default_factory=dict, description="metric -> lowest accepted value")
    max: dict[str, float] = Field(default_factory=dict, description="metric -> highest accepted value")


class ModelingConfig(BaseConfig):
    identity_exclude: ClassVar[frozenset[str]] = BaseConfig.identity_exclude | {"tracker"}

    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

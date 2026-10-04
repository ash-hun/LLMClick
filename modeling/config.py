"""Config sections every modeling recipe shares: experiment tracking and the validation gate."""

from typing import ClassVar

from pydantic import BaseModel, Field

from core.config.schema import BaseConfig


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

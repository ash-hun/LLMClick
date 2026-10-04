"""Request and response bodies of the API; the config itself is the recipe's schema (core.config.schema.BaseConfig)."""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ConfigSummary(BaseModel):
    valid: bool = True
    recipe: str
    experiment: str
    directory: str
    stages: list[str] = Field(description="Stages a run would execute or reuse, in order")
    config: dict[str, Any]


class JobRequest(BaseModel):
    config_path: str | None = Field(default=None, description="YAML path on the server, relative to the working directory")
    config: dict[str, Any] | None = Field(default=None, description="Inline config with the same shape as the YAML")

    @model_validator(mode="after")
    def _one_source(self) -> "JobRequest":
        if (self.config_path is None) == (self.config is None):
            raise ValueError("Give exactly one of config_path or config")
        return self


class JobResponse(BaseModel):
    job_id: str
    status: str
    recipe: str
    experiment: str
    directory: str
    stages: list[str]
    progress: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None

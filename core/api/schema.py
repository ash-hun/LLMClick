"""Request and response bodies of the API; the pipeline config itself is core.config.schema.PipelineConfig."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from core.config.schema import STAGES


class ConfigSummary(BaseModel):
    valid: bool = True
    experiment: str
    directory: str
    config: dict[str, Any]


class JobRequest(BaseModel):
    config_path: str | None = Field(default=None, description="YAML path on the server, relative to the working directory")
    config: dict[str, Any] | None = Field(default=None, description="Inline config with the same shape as the YAML")
    stages: list[str] | None = Field(default=None, description=f"Subset of {list(STAGES)}; default all")

    @model_validator(mode="after")
    def _one_source(self) -> "JobRequest":
        if (self.config_path is None) == (self.config is None):
            raise ValueError("Give exactly one of config_path or config")
        unknown = [s for s in self.stages or [] if s not in STAGES]
        if unknown:
            raise ValueError(f"Unknown stages {unknown}; choose from {list(STAGES)}")
        return self


class JobResponse(BaseModel):
    job_id: str
    status: str
    experiment: str
    directory: str
    stages: list[str]
    result: dict[str, Any] | None = None
    error: str | None = None

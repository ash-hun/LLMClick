"""Config Channel: load a YAML from the server or validate an inline config; nothing is written."""

from typing import Any

from fastapi import APIRouter, HTTPException

from core.api.schema import ConfigSummary
from core.config.experiment import Experiment, load_config
from core.config.schema import PipelineConfig

config_channel_router = APIRouter(prefix="/api/config", tags=["Config Channel"])


def summary(config: PipelineConfig) -> ConfigSummary:
    experiment = Experiment(config)
    return ConfigSummary(experiment=experiment.key, directory=str(experiment.root), config=config.model_dump(mode="json"))


@config_channel_router.post("/load", response_model=ConfigSummary)
def load(config_path: str) -> ConfigSummary:
    try:
        return summary(load_config(config_path))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No such config: {config_path}") from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@config_channel_router.post("/validate", response_model=ConfigSummary)
def validate(config: dict[str, Any]) -> ConfigSummary:
    try:
        pipeline = config.pop("pipeline", {})
        return summary(PipelineConfig(**{**pipeline, **config}))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None

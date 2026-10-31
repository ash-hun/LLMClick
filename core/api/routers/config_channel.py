"""Config Channel: load a YAML from the server or validate an inline config; nothing is written."""

from typing import Annotated, Any

import yaml
from fastapi import APIRouter, Body, HTTPException, Query

from core.api.access_guard import confined
from core.api.api_schema import EXAMPLE_CONFIG, ConfigSummary
from core.pipeline_runner import Pipeline
from core import pipeline_runner

config_channel_router = APIRouter(prefix="/api/config", tags=["Config Channel"])


def parse(config_path: str | None, config: dict[str, Any] | None, **options: Any) -> Pipeline[Any]:
    """The pipeline for a config given by path or inline; 404 for a missing file, 422 for an invalid config, 403 when
    the file, or a path the config names, lies outside `API_PATHS`."""
    try:
        if config_path is not None:
            confined(config_path, "config_path")
            built = pipeline_runner.load(config_path, **options)
        else:
            built = pipeline_runner.build(config or {}, **options)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No such config: {config_path}") from None
    except (ValueError, yaml.YAMLError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    for path in built.config.paths():
        confined(path, "config path")
    return built


def summary(built: Pipeline[Any]) -> ConfigSummary:
    return ConfigSummary(recipe=built.kind, experiment=built.experiment.key, directory=str(built.experiment.root),
                         stages=built.plan(), config=built.config.model_dump(mode="json"))


@config_channel_router.post("/load", response_model=ConfigSummary, summary="Check a config file on the server",
                            description="The experiment key and stage plan of a YAML under API_PATHS. Nothing runs or is written.")
def load(config_path: Annotated[str, Query(description="YAML path on the server, relative to its working directory",
                                           examples=["configs/llm/sft.yaml"])]) -> ConfigSummary:
    return summary(parse(config_path, None))


@config_channel_router.post("/validate", response_model=ConfigSummary, summary="Check an inline config",
                            description="The same check as /load for a config sent in the body, in the shape of the YAML.")
def validate(config: Annotated[dict[str, Any], Body(description="A config in the shape of the YAML", examples=[EXAMPLE_CONFIG])]) -> ConfigSummary:
    return summary(parse(None, config))

"""Job Channel: run the pipeline for a config in the background and poll its status."""

from typing import Any

from fastapi import APIRouter, HTTPException

from core.api import store
from core.api.schema import JobRequest, JobResponse
from core.config.experiment import Experiment, load_config
from core.config.schema import STAGES, PipelineConfig

job_channel_router = APIRouter(prefix="/api/jobs", tags=["Job Channel"])


def parse(request: JobRequest) -> PipelineConfig:
    try:
        if request.config_path is not None:
            return load_config(request.config_path)
        raw: dict[str, Any] = dict(request.config or {})
        pipeline = raw.pop("pipeline", {})
        return PipelineConfig(**{**pipeline, **raw})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No such config: {request.config_path}") from None
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None


@job_channel_router.post("", response_model=JobResponse, status_code=202)
def create(request: JobRequest) -> JobResponse:
    from core import pipeline
    config = parse(request)
    experiment = Experiment(config)
    stages = request.stages or list(STAGES)
    identifier = store.job_id(experiment.key, request.stages)
    job = store.submit(identifier, lambda: pipeline.run(config, request.stages),
                       {"experiment": experiment.key, "directory": str(experiment.root), "stages": stages})
    return JobResponse(**job)


@job_channel_router.get("", response_model=list[JobResponse])
def index() -> list[JobResponse]:
    return [JobResponse(**job) for job in store.all_jobs()]


@job_channel_router.get("/{job_id}", response_model=JobResponse)
def show(job_id: str) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
    return JobResponse(**job)

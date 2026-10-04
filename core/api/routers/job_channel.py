"""Job Channel: run the pipeline for a config in the background and poll its status and progress."""

from typing import Any

from fastapi import APIRouter, HTTPException

from core.api.routers.config_channel import parse
from core.api.schema import JobRequest, JobResponse
from core.progress import StateProgress
from core.api import store

job_channel_router = APIRouter(prefix="/api/jobs", tags=["Job Channel"])


def response(job: dict[str, Any]) -> JobResponse:
    return JobResponse(**{**job, "progress": job["progress"].snapshot()})


@job_channel_router.post("", response_model=JobResponse, status_code=202)
def create(request: JobRequest) -> JobResponse:
    progress = StateProgress()
    built = parse(request.config_path, request.config, progress=progress)
    plan = built.plan()
    job = store.submit(store.job_id(built.experiment.key, plan), built.run,
                       {"recipe": built.kind, "experiment": built.experiment.key,
                        "directory": str(built.experiment.root), "stages": plan, "progress": progress})
    return response(job)


@job_channel_router.get("", response_model=list[JobResponse])
def index() -> list[JobResponse]:
    return [response(job) for job in store.all_jobs()]


@job_channel_router.get("/{job_id}", response_model=JobResponse)
def show(job_id: str) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
    return response(job)

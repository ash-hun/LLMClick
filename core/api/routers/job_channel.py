"""Job Channel: run the pipeline for a config in the background and poll its status and progress."""

from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query

from core.api.routers.config_channel import parse
from core.api.schema import JobRequest, JobResponse
from core.progress import StateProgress
from core.api.store import ACTIVE, CANCELLED
from core.api.store import job_id as job_identifier
from core.api.store import store

Status = Literal["pending", "running", "done", "failed", "cancelled", "interrupted"]

job_channel_router = APIRouter(prefix="/api/jobs", tags=["Job Channel"])


def response(job: dict[str, Any]) -> JobResponse:
    return JobResponse(**{key: value for key, value in job.items() if key in JobResponse.model_fields})


@job_channel_router.post("", response_model=JobResponse, status_code=202)
def create(request: JobRequest) -> JobResponse:
    progress = StateProgress()
    built = parse(request.config_path, request.config, progress=progress)
    plan = built.plan()
    job = store().submit(job_identifier(built.experiment.key, plan), built.run,
                         {"recipe": built.kind, "experiment": built.experiment.key,
                          "directory": str(built.experiment.root), "stages": plan}, progress)
    return response(job)


@job_channel_router.get("", response_model=list[JobResponse])
def index(status: Annotated[Status | None, Query(description="Only jobs in this status")] = None,
          limit: Annotated[int | None, Query(ge=1, description="Only the most recent jobs")] = None) -> list[JobResponse]:
    return [response(job) for job in store().all(status, limit)]


@job_channel_router.get("/{job_id}", response_model=JobResponse)
def show(job_id: str) -> JobResponse:
    job = store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
    return response(job)


@job_channel_router.delete("/{job_id}", response_model=JobResponse, status_code=202,
                           summary="Cancel a pending or running job")
def cancel(job_id: str) -> JobResponse:
    """The job stops at its next progress report and is then `cancelled`; poll it to see that. What its stages
    built so far stays, so submitting the same config again continues from there. 409 for a finished job."""
    job = store().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
    if job["status"] not in ACTIVE and job["status"] != CANCELLED:
        raise HTTPException(status_code=409, detail=f"Job {job_id} is {job['status']}; only a pending or running job can be cancelled")
    cancelled = store().cancel(job_id)
    return response(cancelled or job)

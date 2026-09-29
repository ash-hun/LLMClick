"""In-memory job store: the job id is the experiment key plus the requested stages, so resubmitting is a lookup."""

import threading
import traceback
from collections.abc import Callable
from typing import Any

from core.utils.files import sha256_json

# ponytail: one process, jobs vanish on restart; move to a table when several API replicas share work.
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def job_id(experiment_key: str, stages: list[str] | None) -> str:
    return f"{experiment_key}-{sha256_json(stages or 'all')[:6]}"


def submit(identifier: str, work: Callable[[], Any], detail: dict[str, Any]) -> dict[str, Any]:
    """Start `work` in a thread unless a job with this id is pending, running or done; then return the existing one."""
    with _lock:
        existing = _jobs.get(identifier)
        if existing and existing["status"] in {"pending", "running", "done"}:
            return existing
        job = {"job_id": identifier, "status": "pending", "result": None, "error": None, **detail}
        _jobs[identifier] = job

    def target() -> None:
        job["status"] = "running"
        try:
            job["result"] = work()
            job["status"] = "done"
        except Exception as error:  # noqa: BLE001 - the job record is the error report
            job["error"] = f"{error}\n{traceback.format_exc()}"
            job["status"] = "failed"

    threading.Thread(target=target, name=identifier, daemon=True).start()
    return job


def get(identifier: str) -> dict[str, Any] | None:
    return _jobs.get(identifier)


def all_jobs() -> list[dict[str, Any]]:
    return list(_jobs.values())

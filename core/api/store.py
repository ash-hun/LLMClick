"""Job store: a SQLite table, so jobs survive a restart, and a pool of workers that each own one accelerator."""

import json
import queue
import sqlite3
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from core.settings import get_settings
from core.progress import Cancelled, StateProgress
from core.utils.files import sha256_json
from core.utils import device

ACTIVE = {"pending", "running"}
INTERRUPTED = "interrupted"
CANCELLED = "cancelled"


def job_id(experiment_key: str, stages: list[str]) -> str:
    return f"{experiment_key}-{sha256_json(stages)[:6]}"


class JobStore:
    """The table is the record of every job; live progress of running jobs is read from memory on top of it."""

    def __init__(self, path: Path, workers: int = 1) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.live: dict[str, StateProgress] = {}
        self.database = sqlite3.connect(path, check_same_thread=False)
        with self.lock, self.database:
            self.database.execute("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
        # Jobs that were queued or running when the last server stopped did not finish: say so, so that a
        # resubmission runs them again (the pipeline resumes from what is already built).
        for record in self.all():
            if record["status"] in ACTIVE:
                self.save({**record, "status": INTERRUPTED, "error": "the server stopped before this job finished; submit it again"})
        self.workers = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self.slots: queue.Queue[int] = queue.Queue()
        for index in range(workers):
            self.slots.put(index)

    def save(self, record: dict[str, Any]) -> None:
        with self.lock, self.database:
            self.database.execute("INSERT INTO jobs (job_id, record) VALUES (?, ?) ON CONFLICT(job_id) DO UPDATE SET record = excluded.record",
                                  (record["job_id"], json.dumps(record, ensure_ascii=False, default=str)))

    def read(self, row: tuple[str]) -> dict[str, Any]:
        record: dict[str, Any] = json.loads(row[0])
        progress = self.live.get(record["job_id"])
        return {**record, "progress": progress.snapshot()} if progress is not None else record

    def get(self, identifier: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.database.execute("SELECT record FROM jobs WHERE job_id = ?", (identifier,)).fetchone()
        return self.read(row) if row else None

    def all(self, status: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        """Jobs in submission order; `status` keeps one status, `limit` keeps the most recent ones."""
        with self.lock:
            rows = self.database.execute("SELECT record FROM jobs ORDER BY rowid").fetchall()
        jobs = [job for job in (self.read(row) for row in rows) if status is None or job["status"] == status]
        return jobs[-limit:] if limit else jobs

    def cancel(self, identifier: str) -> dict[str, Any] | None:
        """Ask a pending or running job to stop; it is `cancelled` once the pipeline has stopped at its next progress
        report. A job that is not active is returned as it is."""
        with self.lock:
            progress = self.live.get(identifier)
            if progress is not None:
                progress.cancel()
        return self.get(identifier)

    def submit(self, identifier: str, work: Callable[[], Any], detail: dict[str, Any], progress: StateProgress) -> dict[str, Any]:
        """Queue `work` unless a job with this id is pending or running; then return that one. A finished, failed
        or interrupted job is run again: stages that are still valid return from cache at once, and an edited
        input or a deleted output is rebuilt, exactly as a second CLI run would."""
        with self.lock:
            row = self.database.execute("SELECT record FROM jobs WHERE job_id = ?", (identifier,)).fetchone()
            if row and json.loads(row[0])["status"] in ACTIVE and identifier in self.live:
                return self.read(row)
            self.live[identifier] = progress
        record = {"job_id": identifier, "status": "pending", "result": None, "error": None, "progress": progress.snapshot(), **detail}
        self.save(record)

        def target() -> None:
            index = self.slots.get()  # each worker owns one accelerator for the length of the job
            device.slot.index = index
            self.save({**record, "status": "running", "device_slot": index})
            try:
                progress.check()  # cancelled while it waited for a worker: nothing runs
                outcome = {"status": "done", "result": work()}
            except Cancelled:
                outcome = {"status": CANCELLED}  # what was built stays; a resubmission continues from it
            except Exception as error:  # noqa: BLE001 - the job record is the error report
                outcome = {"status": "failed", "error": f"{error}\n{traceback.format_exc()}"}
            finally:
                device.slot.index = None
                self.slots.put(index)
            self.save({**record, **outcome, "device_slot": index, "progress": progress.snapshot()})
            self.live.pop(identifier, None)

        self.workers.submit(target)
        return self.get(identifier) or record


_store: JobStore | None = None


def configure(path: Path | None = None, workers: int | None = None) -> JobStore:
    """(Re)open the job store; the API calls it lazily with the settings, tests call it with a temporary file."""
    global _store
    settings = get_settings()
    _store = JobStore(path or Path(settings.JOBS_DB), workers or settings.JOB_WORKERS)
    return _store


def store() -> JobStore:
    return _store or configure()

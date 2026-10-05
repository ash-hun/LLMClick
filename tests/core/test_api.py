import time
from pathlib import Path
from typing import Any

import threading

import pytest
from fastapi.testclient import TestClient

from core.api import store as job_store
from core.api.store import JobStore
from core.progress import StateProgress
from core.utils import device
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def temporary_job_table(tmp_path: Path) -> None:
    job_store.configure(tmp_path / "jobs.sqlite", workers=1)


def config(model: Path, out: Path) -> dict[str, Any]:
    return {
        "pipeline": {"recipe": "llm_sft", "name": "api-test", "seed": 7, "output_dir": str(out), "device": "cpu"},
        "model": {"architecture": "transformer", "name": str(model)},
        "data": {"sources": [{"name": "local_jsonl", "path": "samples/llm_sft.jsonl"}], "validation": 0.2},
        "training": {"lr": 1e-3, "batch_size": 4, "max_length": 64, "max_steps": 2},
    }


def test_health_and_system() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    recipes = client.get("/api/system/recipes").json()
    assert recipes["llm_sft"]["stages"] == ["data", "train", "validate"] and "architecture" in recipes["llm_sft"]


def test_validate_and_load(tiny_model: Path, tmp_path: Path) -> None:
    body = config(tiny_model, tmp_path)
    ok = client.post("/api/config/validate", json=body)
    assert ok.status_code == 200 and ok.json()["experiment"].startswith("api-test-")
    assert ok.json()["stages"] == ["data", "train", "validate"]
    bad = {**body, "model": {**body["model"], "architecture": "nope"}}
    assert client.post("/api/config/validate", json=bad).status_code == 422
    assert client.post("/api/config/validate", json={"pipeline": {"name": "x"}}).status_code == 422
    loaded = client.post("/api/config/load", params={"config_path": "configs/llm/sft.yaml"})
    assert loaded.status_code == 200 and loaded.json()["recipe"] == "llm_sft"
    assert client.post("/api/config/load", params={"config_path": "configs/none.yaml"}).status_code == 404


def test_job_runs_the_pipeline_and_resubmit_is_same_job(tiny_model: Path, tmp_path: Path) -> None:
    body = {"config": config(tiny_model, tmp_path)}
    created = client.post("/api/jobs", json=body)
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    assert client.post("/api/jobs", json=body).json()["job_id"] == job_id
    for _ in range(300):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        time.sleep(0.1)
    assert job["status"] == "done", job["error"]
    assert job["result"]["stages"]["train"]["steps"] == 2 and job["result"]["stages"]["validate"]["passed"] is True
    assert job["progress"]["stages"] == {stage: "done" for stage in ["data", "train", "validate"]}
    again = client.post("/api/jobs", json=body).json()  # a finished job runs again; everything comes from cache
    assert again["job_id"] == job_id and again["status"] in {"pending", "running", "done"}
    for _ in range(300):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        time.sleep(0.1)
    assert job["status"] == "done" and job["progress"]["stages"] == {stage: "cached" for stage in ["data", "train", "validate"]}
    assert client.get("/api/jobs/nope").status_code == 404
    assert any(j["job_id"] == job_id for j in client.get("/api/jobs").json())


def wait(store: JobStore, identifier: str) -> dict[str, Any]:
    for _ in range(300):
        job = store.get(identifier)
        if job and job["status"] not in {"pending", "running"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_jobs_survive_a_restart_and_unfinished_ones_are_marked(tmp_path: Path) -> None:
    path = tmp_path / "table.sqlite"
    first = JobStore(path)
    first.submit("done-job", lambda: {"answer": 42}, {"recipe": "r"}, StateProgress())
    assert wait(first, "done-job")["result"] == {"answer": 42}
    first.save({"job_id": "cut-off", "status": "running", "recipe": "r", "result": None, "error": None, "progress": {}})
    reopened = JobStore(path)  # a new server process
    assert reopened.get("done-job")["status"] == "done" and reopened.get("done-job")["result"] == {"answer": 42}
    cut = reopened.get("cut-off")
    assert cut["status"] == "interrupted" and "submit it again" in cut["error"]
    reopened.submit("cut-off", lambda: "resumed", {"recipe": "r"}, StateProgress())
    assert wait(reopened, "cut-off")["result"] == "resumed"
    assert [job["job_id"] for job in reopened.all()] == ["done-job", "cut-off"]


def test_workers_run_jobs_at_once_each_on_its_own_slot(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "table.sqlite", workers=2)
    both_running, slots = threading.Barrier(2, timeout=5), []

    def work() -> int:
        slots.append(device.slot.index)
        both_running.wait()  # passes only if the two jobs overlap
        return device.slot.index

    for name in ("a", "b"):
        store.submit(name, work, {"recipe": "r"}, StateProgress())
    assert sorted(wait(store, name)["result"] for name in ("a", "b")) == [0, 1] and sorted(slots) == [0, 1]
    assert {store.get(name)["device_slot"] for name in ("a", "b")} == {0, 1}


def test_a_failed_job_keeps_its_error_and_frees_its_slot(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "table.sqlite")
    store.submit("bad", lambda: 1 / 0, {"recipe": "r"}, StateProgress())
    assert "ZeroDivisionError" in wait(store, "bad")["error"]
    store.submit("good", lambda: "ok", {"recipe": "r"}, StateProgress())
    assert wait(store, "good")["result"] == "ok"

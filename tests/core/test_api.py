import time
from pathlib import Path
from typing import Any

import threading

import pytest
from fastapi.testclient import TestClient

from config import get_settings
from core.api import store as job_store
from core.api.store import JobStore
from core.progress import StateProgress
from core.utils import device
from main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def temporary_job_table(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    # the tests keep their model and output under pytest's temporary directory, so the API may reach it too
    monkeypatch.setenv("API_PATHS", f".:{tmp_path_factory.getbasetemp()}")
    monkeypatch.delenv("API_TOKEN", raising=False)
    get_settings.cache_clear()
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


DETAIL: dict[str, Any] = {"recipe": "r", "experiment": "e", "directory": "d", "stages": ["train"]}


def test_a_running_job_is_cancelled_at_its_next_progress_report(tmp_path: Path) -> None:
    progress, started = StateProgress(), threading.Event()

    def work() -> None:
        for step in range(10_000):
            progress.update(step, 10_000)  # raises once cancelled
            started.set()
            time.sleep(0.001)

    job_store.configure(tmp_path / "table.sqlite", workers=1)
    store = job_store.store()
    store.submit("long", work, DETAIL, progress)
    assert started.wait(5)
    assert client.delete("/api/jobs/long").status_code == 202
    job = wait(store, "long")
    assert job["status"] == "cancelled" and job["error"] is None and job["progress"]["done"] < 10_000
    assert client.delete("/api/jobs/long").status_code == 202  # cancelling again is harmless
    store.submit("long", lambda: "again", DETAIL, StateProgress())  # a cancelled job can be resubmitted
    assert wait(store, "long")["result"] == "again"
    assert client.delete("/api/jobs/long").status_code == 409 and client.delete("/api/jobs/nope").status_code == 404


def test_a_pending_job_cancelled_before_its_turn_never_runs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "table.sqlite", workers=1)
    release, ran = threading.Event(), []
    store.submit("first", lambda: {"waited": release.wait(5)}, DETAIL, StateProgress())
    store.submit("second", lambda: ran.append(1), DETAIL, StateProgress())
    assert store.get("second")["status"] == "pending"
    store.cancel("second")
    release.set()
    assert wait(store, "second")["status"] == "cancelled" and ran == []
    assert [job["job_id"] for job in store.all(status="cancelled")] == ["second"]
    assert [job["job_id"] for job in store.all(limit=1)] == ["second"]
    job_store.configure(tmp_path / "table.sqlite", workers=1)
    assert [job["job_id"] for job in client.get("/api/jobs", params={"status": "done"}).json()] == ["first"]
    assert client.get("/api/jobs", params={"status": "nope"}).status_code == 422


def test_paths_outside_api_paths_are_refused(tiny_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = config(tiny_model, tmp_path)
    assert client.post("/api/config/validate", json=body).status_code == 200
    outside = str(Path("/").resolve())
    for section, keys in [("pipeline", {"output_dir": outside}), ("model", {"init": f"{tmp_path}/../../escape"}),
                          ("data", {"sources": [{"name": "local_jsonl", "path": "/etc/hosts"}]})]:
        refused = client.post("/api/config/validate", json={**body, section: {**body[section], **keys}})
        assert refused.status_code == 403 and "outside API_PATHS" in refused.json()["detail"], section
    assert client.post("/api/jobs", json={"config": {**body, "pipeline": {**body["pipeline"], "output_dir": outside}}}).status_code == 403
    assert client.post("/api/config/load", params={"config_path": "../../../etc/passwd"}).status_code == 403
    assert client.post("/api/config/load", params={"config_path": "/etc/passwd"}).status_code == 403
    monkeypatch.setenv("API_PATHS", "configs")  # rows under samples/ are then out of reach, the config file is not
    get_settings.cache_clear()
    assert client.post("/api/config/load", params={"config_path": "configs/llm/sft.yaml"}).status_code == 403


def test_api_token_guards_every_api_route_but_not_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_TOKEN", "s3cret")
    get_settings.cache_clear()
    assert client.get("/health").status_code == 200
    for request in (lambda h: client.get("/api/jobs", headers=h), lambda h: client.get("/api/system/recipes", headers=h),
                    lambda h: client.post("/api/config/validate", json={}, headers=h)):
        assert request({}).status_code == 401 and request({"Authorization": "Bearer wrong"}).status_code == 401
        assert request({"Authorization": "Bearer s3cret"}).status_code != 401
    assert client.get("/api/jobs", headers={"Authorization": "Bearer s3cret"}).json() == []

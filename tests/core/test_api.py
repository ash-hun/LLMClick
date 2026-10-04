import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


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

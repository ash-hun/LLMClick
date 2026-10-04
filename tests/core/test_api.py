import time

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)
STAGES = ["data", "benchmarks", "mix"]


def test_health_and_system() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    recipes = client.get("/api/system/recipes").json()
    assert recipes["jev"]["stages"][0] == "data" and "builder" in recipes["jev"]


def test_validate_and_load(local_config: dict) -> None:
    ok = client.post("/api/config/validate", json=local_config)
    assert ok.status_code == 200 and ok.json()["experiment"].startswith("local-test-")
    assert ok.json()["stages"][-2:] == ["validate", "evaluate"]
    bad = {**local_config, "model": {**local_config["model"], "backbone": "nope"}}
    assert client.post("/api/config/validate", json=bad).status_code == 422
    assert client.post("/api/config/validate", json={"pipeline": {"name": "x"}}).status_code == 422
    loaded = client.post("/api/config/load", params={"config_path": "configs/jeff_public_only.yaml"})
    assert loaded.status_code == 200 and loaded.json()["recipe"] == "jev"
    assert client.post("/api/config/load", params={"config_path": "configs/none.yaml"}).status_code == 404


def test_job_runs_local_stages_and_resubmit_is_same_job(local_config: dict) -> None:
    body = {"config": {**local_config, "pipeline": {**local_config["pipeline"], "stages": STAGES}}}
    created = client.post("/api/jobs", json=body)
    assert created.status_code == 202
    job_id = created.json()["job_id"]
    assert client.post("/api/jobs", json=body).json()["job_id"] == job_id
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        time.sleep(0.1)
    assert job["status"] == "done", job["error"]
    assert job["result"]["stages"]["mix"]["sizes"]["public"] == 45
    assert job["progress"]["stages"] == {stage: "done" for stage in ["data", "benchmarks", "synthetic", "mix"]}
    assert client.get("/api/jobs/nope").status_code == 404
    assert any(j["job_id"] == job_id for j in client.get("/api/jobs").json())

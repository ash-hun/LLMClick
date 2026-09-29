import time

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_and_system() -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert "builder" in client.get("/api/system/registries").json()
    assert client.get("/api/system/stages").json()[0] == "data"


def test_validate_and_load(local_config: dict) -> None:
    ok = client.post("/api/config/validate", json=local_config)
    assert ok.status_code == 200 and ok.json()["experiment"].startswith("local-test-")
    bad = client.post("/api/config/validate", json={"pipeline": {"name": "x"}, "model": {"backbone": "nope", "name": "m", "revision": "0" * 40}})
    assert bad.status_code == 422
    loaded = client.post("/api/config/load", params={"config_path": "configs/jeff_public_only.yaml"})
    assert loaded.status_code == 200
    assert client.post("/api/config/load", params={"config_path": "configs/none.yaml"}).status_code == 404


def test_job_runs_local_stages_and_resubmit_is_same_job(local_config: dict) -> None:
    body = {"config": local_config, "stages": ["data", "benchmarks", "mix"]}
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
    assert job["result"]["stages"]["mix"]["sizes"]["public"] == 50
    assert client.get("/api/jobs/nope").status_code == 404
    assert any(j["job_id"] == job_id for j in client.get("/api/jobs").json())

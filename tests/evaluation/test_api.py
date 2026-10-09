"""The evaluation reports under an output directory, listed and shown through the API, and sent to the tracker."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core import pipeline
from core.api.app import app
from core.settings import get_settings
from modeling.tracker import flatten
from tests.evaluation.test_custom import evaluation, trained  # noqa: F401
from tests.modeling.tuning.test_recipes import FakeWandb, wandb  # noqa: F401


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("API_PATHS", f".:{tmp_path.parent.parent}")
    monkeypatch.delenv("API_TOKEN", raising=False)
    get_settings.cache_clear()
    return TestClient(app)


def test_reports_are_listed_filtered_and_shown(trained: dict[str, Any], tmp_path: Path, client: TestClient) -> None:  # noqa: F811
    held_out = trained["stages"]["data"]["validation"]
    done = pipeline.build(evaluation(trained["directory"], held_out, tmp_path)).run()
    pipeline.build({**evaluation(trained["directory"], held_out, tmp_path, "base"), "pipeline": {"recipe": "evaluation_custom", "name": "base-eval", "output_dir": str(tmp_path), "device": "cpu"}}).run()
    listed = client.get("/api/evaluations", params={"output_dir": str(tmp_path)}).json()
    assert [entry["recipe"] for entry in listed] == ["evaluation_custom", "evaluation_custom"]
    first = next(entry for entry in listed if entry["experiment"] == done["experiment"])
    assert first["scores"] == pytest.approx(done["stages"]["report"]["scores"]) and first["model"]["experiment"] == trained["directory"]
    assert first["contamination"]["items"] == 16 and first["report"].endswith("report.json")
    assert len(client.get("/api/evaluations", params={"output_dir": str(tmp_path), "experiment": trained["directory"]}).json()) == 2
    assert client.get("/api/evaluations", params={"output_dir": str(tmp_path), "experiment": "output/nowhere"}).json() == []
    assert client.get("/api/evaluations", params={"output_dir": str(tmp_path), "recipe": "evaluation_benchmark"}).json() == []
    shown = client.get(f"/api/evaluations/{done['experiment']}", params={"output_dir": str(tmp_path)}).json()
    assert shown["report"]["scores"] == first["scores"] and shown["report"]["model"]["recipe"] == "llm_sft"
    assert client.get("/api/evaluations/nope", params={"output_dir": str(tmp_path)}).status_code == 404
    assert client.get("/api/evaluations", params={"output_dir": "/"}).status_code == 403
    assert client.get("/api/evaluations", params={"output_dir": str(tmp_path / "empty")}).json() == []


def test_the_report_goes_to_the_tracker_once(trained: dict[str, Any], tmp_path: Path, wandb: FakeWandb) -> None:  # noqa: F811
    body = {**evaluation(trained["directory"], trained["stages"]["data"]["validation"], tmp_path), "tracker": {"enabled": True, "project": "p"}}
    result = pipeline.build(body).run()
    assert len(wandb.runs) == 1 and wandb.runs[0]["project"] == "p" and wandb.finished == 1
    numbers, table = wandb.reports
    assert numbers["scores/loss"] == pytest.approx(result["stages"]["report"]["scores"]["loss"]) and "contamination/exact" in numbers
    assert table["report"].columns == ["name", "value"] and dict(table["report"].data)["scores/loss"] == numbers["scores/loss"]
    pipeline.build(body).run()  # cached report: already sent to this project
    assert len(wandb.runs) == 1
    assert flatten({"a": {"b": 1, "settings": {"c": 2}}, "d": True, "e": "text"}) == {"a/b": 1.0, "d": 1.0}

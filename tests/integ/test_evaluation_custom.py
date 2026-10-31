"""`evaluation_custom` on a tiny trained experiment: the recipe's metrics on any rows and checkpoints, contamination, listing, tracker."""

import json
from typing import Any
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.recipe_fakes import FakeWandb
from core.progress_reporter import StateProgress
from core.settings import get_settings
from modeling.metric_tracker import flatten
from core.api.api_app import app
from core import pipeline_runner


def evaluation(experiment: str, rows: str, out: Path, checkpoint: str = "validate", **keys: Any) -> dict[str, Any]:
    return {"pipeline": {"recipe": "evaluation_custom", "name": "eval", "output_dir": str(out), "device": "cpu"},
            "model": {"experiment": experiment, "checkpoint": checkpoint},
            "data": {"sources": [{"name": "local_jsonl", "path": rows}]}, "evaluation": {"batch_size": 8}, **keys}


def test_scores_a_checkpoint_on_rows_with_the_recipes_own_metrics(trained: dict[str, Any], tmp_path: Path) -> None:
    held_out = trained["stages"]["data"]["validation"]  # the rows validate measured: the same numbers must come out
    result = pipeline_runner.build(evaluation(trained["directory"], held_out, tmp_path)).run()
    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    assert report["scores"] == pytest.approx(trained["stages"]["validate"]["metrics"])
    assert report["model"] == {"experiment": trained["directory"], "checkpoint": f"{trained['directory']}/validate/checkpoint",
                               "recipe": "llm_sft", "name": None, "revision": None, "architecture": None}
    assert report["rows"]["scored"] == 16 and report["rows"]["skipped"] == 0
    overlap = report["contamination"]  # held-out rows: none of them trained, so no exact match; short rows may share n-grams
    assert overlap["items"] == 16 and overlap["exact"] == 0 and overlap["training_rows"] == 64 and overlap["ngram"] == 8
    assert report["settings"] == {"batch_size": 8, "max_length": 64, "device": "cpu"}
    progress = StateProgress()
    again = pipeline_runner.build(evaluation(trained["directory"], held_out, tmp_path), progress).run()
    assert json.dumps(again) == json.dumps(result)
    assert progress.snapshot()["stages"] == {"rows": "cached", "score": "cached", "report": "cached"}


def test_base_and_train_checkpoints_are_other_models(trained: dict[str, Any], tmp_path: Path) -> None:
    held_out = trained["stages"]["data"]["validation"]
    scores = {}
    for checkpoint in ("validate", "train", "base"):
        result = pipeline_runner.build(evaluation(trained["directory"], held_out, tmp_path, checkpoint)).run()
        scores[checkpoint] = result["stages"]["report"]["scores"]["loss"]
    assert scores["validate"] == scores["train"]  # the same weights, found through the other link
    assert scores["base"] != scores["validate"]  # the weights the experiment started from
    explicit = evaluation(trained["directory"], held_out, tmp_path, trained["stages"]["train"]["checkpoint"])
    assert pipeline_runner.build(explicit).run()["stages"]["report"]["scores"]["loss"] == scores["train"]


def test_rows_are_checked_against_the_recipe_and_missing_experiments_fail_at_run_time(trained: dict[str, Any],
                                                                                        tmp_path: Path) -> None:
    built = pipeline_runner.build(evaluation(trained["directory"], "samples/llm_dpo.jsonl", tmp_path))
    with pytest.raises(ValueError, match="Row 0 does not fit"):
        built.run()
    missing = pipeline_runner.build(evaluation(str(tmp_path / "nowhere"), "samples/llm_sft.jsonl", tmp_path))  # validates
    assert missing.plan() == ["rows", "score", "report"]
    with pytest.raises(FileNotFoundError, match="run that experiment first"):
        missing.run()
    unpassed = pipeline_runner.build(evaluation(trained["directory"], "samples/llm_sft.jsonl", tmp_path, str(tmp_path / "no-dir")))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        unpassed.run()


def test_the_api_knows_the_recipe_and_confines_its_paths(trained: dict[str, Any], tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PATHS", f".:{tmp_path.parent.parent}")
    get_settings.cache_clear()
    client = TestClient(app)
    recipes = client.get("/api/system/recipes").json()
    assert recipes["evaluation_custom"]["stages"] == ["rows", "score", "report"]
    assert "local_jsonl" in recipes["evaluation_custom"]["source"] and "base" in recipes["evaluation_custom"]["checkpoint"]
    body = evaluation(trained["directory"], "samples/llm_sft.jsonl", tmp_path)
    assert client.post("/api/config/validate", json=body).status_code == 200
    outside = {**body, "model": {**body["model"], "experiment": "/etc"}}
    assert client.post("/api/config/validate", json=outside).status_code == 403


def test_shipped_evaluation_config_points_at_the_sft_experiment() -> None:
    sft = pipeline_runner.load("configs/llm/sft.yaml")
    custom = pipeline_runner.load("configs/evaluation/custom.yaml")
    assert custom.config.model.experiment == f"output/{sft.experiment.key}"  # edit one config, update the other


def test_shipped_decision_and_embedding_configs_point_at_their_experiments() -> None:
    for config, source in [("configs/evaluation/decision.yaml", "configs/llm/decision_pointer.yaml"),
                           ("configs/evaluation/embedding.yaml", "configs/embedding/contrastive.yaml")]:
        trained = pipeline_runner.load(source)
        assert pipeline_runner.load(config).config.model.experiment == f"output/{trained.experiment.key}"


def test_base_checkpoints_report_no_contamination(trained: dict[str, Any], tmp_path: Path) -> None:
    body = evaluation(trained["directory"], trained["stages"]["data"]["validation"], tmp_path, "base")
    result = pipeline_runner.build(body).run()
    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    assert report["contamination"] is None  # the base weights never saw the experiment's rows
    contaminated = evaluation(trained["directory"], trained["stages"]["data"]["train"], tmp_path)  # the rows it trained on
    report = json.loads(Path(pipeline_runner.build(contaminated).run()["stages"]["report"]["report"]).read_text())
    assert report["contamination"]["exact"] == 64 and report["contamination"]["near"] == 64


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("API_PATHS", f".:{tmp_path.parent.parent}")
    monkeypatch.delenv("API_TOKEN", raising=False)
    get_settings.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()


def test_reports_are_listed_filtered_and_shown(trained: dict[str, Any], tmp_path: Path, client: TestClient) -> None:
    held_out = trained["stages"]["data"]["validation"]
    done = pipeline_runner.build(evaluation(trained["directory"], held_out, tmp_path)).run()
    pipeline_runner.build({**evaluation(trained["directory"], held_out, tmp_path, "base"), "pipeline": {"recipe": "evaluation_custom", "name": "base-eval", "output_dir": str(tmp_path), "device": "cpu"}}).run()
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


def test_the_report_goes_to_the_tracker_once(trained: dict[str, Any], tmp_path: Path, wandb: FakeWandb) -> None:
    body = {**evaluation(trained["directory"], trained["stages"]["data"]["validation"], tmp_path), "tracker": {"enabled": True, "project": "p"}}
    result = pipeline_runner.build(body).run()
    assert len(wandb.runs) == 1 and wandb.runs[0]["project"] == "p" and wandb.finished == 1
    numbers, table = wandb.reports
    assert numbers["scores/loss"] == pytest.approx(result["stages"]["report"]["scores"]["loss"]) and "contamination/exact" in numbers
    assert table["report"].columns == ["name", "value"] and dict(table["report"].data)["scores/loss"] == numbers["scores/loss"]
    pipeline_runner.build(body).run()  # cached report: already sent to this project
    assert len(wandb.runs) == 1
    assert flatten({"a": {"b": 1, "settings": {"c": 2}}, "d": True, "e": "text"}) == {"a/b": 1.0, "d": 1.0}

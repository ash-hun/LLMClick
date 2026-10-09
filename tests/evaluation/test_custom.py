"""`evaluation_custom` on a tiny trained experiment: the recipe's metrics on any rows, for any of its checkpoints."""

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core import pipeline
from core.api.app import app
from core.progress import StateProgress
from tests.modeling.tuning.test_recipes import raw


def evaluation(experiment: str, rows: str, out: Path, checkpoint: str = "validate", **keys: Any) -> dict[str, Any]:
    return {"pipeline": {"recipe": "evaluation_custom", "name": "eval", "output_dir": str(out), "device": "cpu"},
            "model": {"experiment": experiment, "checkpoint": checkpoint},
            "data": {"sources": [{"name": "local_jsonl", "path": rows}]}, "evaluation": {"batch_size": 8}, **keys}


@pytest.fixture(scope="module")
def trained(tiny_model: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("trained")
    result = pipeline.build(raw("llm_sft", tiny_model, out)).run()
    return result


def test_scores_a_checkpoint_on_rows_with_the_recipes_own_metrics(trained: dict[str, Any], tmp_path: Path) -> None:
    held_out = trained["stages"]["data"]["validation"]  # the rows validate measured: the same numbers must come out
    result = pipeline.build(evaluation(trained["directory"], held_out, tmp_path)).run()
    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    assert report["scores"] == pytest.approx(trained["stages"]["validate"]["metrics"])
    assert report["model"] == {"experiment": trained["directory"], "checkpoint": f"{trained['directory']}/validate/checkpoint",
                               "recipe": "llm_sft"}
    assert report["rows"]["scored"] == 16 and report["rows"]["skipped"] == 0
    overlap = report["contamination"]  # held-out rows: none of them trained, so no exact match; short rows may share n-grams
    assert overlap["items"] == 16 and overlap["exact"] == 0 and overlap["training_rows"] == 64 and overlap["ngram"] == 8
    assert report["settings"] == {"batch_size": 8, "max_length": 64, "device": "cpu"}
    progress = StateProgress()
    again = pipeline.build(evaluation(trained["directory"], held_out, tmp_path), progress).run()
    assert json.dumps(again) == json.dumps(result)
    assert progress.snapshot()["stages"] == {"rows": "cached", "score": "cached", "report": "cached"}


def test_base_and_train_checkpoints_are_other_models(trained: dict[str, Any], tmp_path: Path) -> None:
    held_out = trained["stages"]["data"]["validation"]
    scores = {}
    for checkpoint in ("validate", "train", "base"):
        result = pipeline.build(evaluation(trained["directory"], held_out, tmp_path, checkpoint)).run()
        scores[checkpoint] = result["stages"]["report"]["scores"]["loss"]
    assert scores["validate"] == scores["train"]  # the same weights, found through the other link
    assert scores["base"] != scores["validate"]  # the weights the experiment started from
    explicit = evaluation(trained["directory"], held_out, tmp_path, trained["stages"]["train"]["checkpoint"])
    assert pipeline.build(explicit).run()["stages"]["report"]["scores"]["loss"] == scores["train"]


def test_rows_are_checked_against_the_recipe_and_missing_experiments_fail_at_run_time(trained: dict[str, Any],
                                                                                        tmp_path: Path) -> None:
    built = pipeline.build(evaluation(trained["directory"], "samples/llm_dpo.jsonl", tmp_path))
    with pytest.raises(ValueError, match="Row 0 does not fit"):
        built.run()
    missing = pipeline.build(evaluation(str(tmp_path / "nowhere"), "samples/llm_sft.jsonl", tmp_path))  # validates
    assert missing.plan() == ["rows", "score", "report"]
    with pytest.raises(FileNotFoundError, match="run that experiment first"):
        missing.run()
    unpassed = pipeline.build(evaluation(trained["directory"], "samples/llm_sft.jsonl", tmp_path, str(tmp_path / "no-dir")))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        unpassed.run()


def test_the_api_knows_the_recipe_and_confines_its_paths(trained: dict[str, Any], tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    from core.settings import get_settings

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
    sft = pipeline.load("configs/llm/sft.yaml")
    custom = pipeline.load("configs/evaluation/custom.yaml")
    assert custom.config.model.experiment == f"output/{sft.experiment.key}"  # edit one config, update the other


def test_shipped_decision_and_embedding_configs_point_at_their_experiments() -> None:
    for config, source in [("configs/evaluation/decision.yaml", "configs/llm/decision_pointer.yaml"),
                           ("configs/evaluation/embedding.yaml", "configs/embedding/contrastive.yaml")]:
        trained = pipeline.load(source)
        assert pipeline.load(config).config.model.experiment == f"output/{trained.experiment.key}"

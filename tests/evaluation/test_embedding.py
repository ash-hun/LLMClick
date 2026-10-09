"""`evaluation_embedding`: the contrastive checkpoint through mteb, on its mock tasks (no download)."""

import json
from pathlib import Path
from typing import Any

import pytest

from core import pipeline
from core.progress import StateProgress
from evaluation.embedding import Encoder, model_meta
from evaluation.source import SourceExperiment
from evaluation.config import SourceModel
from tests.modeling.tuning.test_recipes import raw


@pytest.fixture(scope="module")
def embedding(tiny_model: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("embedding")
    return pipeline.build(raw("embedding_contrastive", tiny_model, out)).run()


def config(experiment: str, out: Path, tasks: list[str], **keys: Any) -> dict[str, Any]:
    return {"pipeline": {"recipe": "evaluation_embedding", "name": "emb", "output_dir": str(out), "device": "cpu"},
            "model": {"experiment": experiment, "checkpoint": "validate"}, "tasks": tasks, "evaluation": {"batch_size": 4}, **keys}


def test_the_encoder_passes_mteb_mock_text_tasks(embedding: dict[str, Any]) -> None:
    import mteb
    from mteb import TaskResult

    source = SourceExperiment(SourceModel(experiment=embedding["directory"], checkpoint="validate"))
    backbone, _, trained = source.build()
    backbone.load("cpu", source.checkpoint())
    try:
        outcome = mteb.mock_run(Encoder(backbone, 64, model_meta(mteb, source, 64)))  # type: ignore[arg-type]
    finally:
        backbone.release()
    results = dict(outcome.items()) if hasattr(outcome, "items") else dict(outcome)
    ran = {name: value for name, value in results.items() if value is not None}
    for name in ("MockSTSTask", "MockRetrievalTask", "MockClassificationTask", "MockPairClassificationTask"):
        assert isinstance(ran[name], TaskResult), (name, ran[name])


def test_one_stage_per_task_with_the_main_score(embedding: dict[str, Any], tmp_path: Path) -> None:
    progress = StateProgress()
    result = pipeline.build(config(embedding["directory"], tmp_path, ["MockSTSTask", "MockRetrievalTask"]), progress).run()
    assert progress.snapshot()["stages"] == {"score:MockSTSTask": "done", "score:MockRetrievalTask": "done", "report": "done"}
    report = json.loads(Path(result["stages"]["report"]["report"]).read_text())
    sts = report["tasks"]["MockSTSTask"]
    assert sts["metric"] == "cosine_spearman" and -1 <= sts["main_score"] <= 1 and sts["type"] == "STS" and Path(sts["results"]).exists()
    assert report["settings"]["package"] == {"name": "mteb", "version": report["settings"]["package"]["version"]} and report["model"]["recipe"] == "embedding_contrastive"
    again = StateProgress()
    pipeline.build(config(embedding["directory"], tmp_path, ["MockSTSTask", "MockRetrievalTask", "MockClassificationTask"]), again).run()
    assert again.snapshot()["stages"]["score:MockSTSTask"] == "cached" and again.snapshot()["stages"]["score:MockClassificationTask"] == "done"
    with pytest.raises(ValueError, match="unknown MTEB task"):
        pipeline.build(config(embedding["directory"], tmp_path, ["NoSuchTask"])).run()


def test_a_non_embedding_experiment_is_refused(tiny_model: Path, tmp_path: Path) -> None:
    sft = pipeline.build(raw("llm_sft", tiny_model, tmp_path)).run()
    with pytest.raises(ValueError, match="not an embedding experiment"):
        pipeline.build(config(sft["directory"], tmp_path, ["MockSTSTask"])).run()

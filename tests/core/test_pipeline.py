"""data -> benchmarks -> mix on local fixtures, twice: the second run must change nothing."""

import json
from pathlib import Path

from core import pipeline
from core.config.schema import PipelineConfig
from core.modules.tuning import trainer
from core.utils.files import sha256_file

STAGES = ["data", "benchmarks", "mix"]


def build(raw: dict) -> PipelineConfig:
    return PipelineConfig(**raw["pipeline"], **{k: v for k, v in raw.items() if k != "pipeline"})


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "pipeline.log"}


def test_local_stages_run_and_are_idempotent(local_config: dict) -> None:
    config = build(local_config)
    first = pipeline.run(config, STAGES)
    root = Path(first["directory"])
    assert first["stages"]["data"]["rows"] == {"train": 50, "dev": 6, "temperature": 4}
    assert (root / "data" / "mix" / "public.jsonl").exists()
    assert first["stages"]["mix"]["leaks"]["public"] == 0
    before = snapshot(root)
    second = pipeline.run(config, STAGES)
    assert snapshot(root) == before
    assert second["stages"] == first["stages"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest["stages"]) == set(STAGES) | {"synthetic"}  # mix depends on synthetic, which records "disabled" as {}


def test_changed_config_is_a_new_experiment(local_config: dict) -> None:
    config = build(local_config)
    pipeline.run(config, STAGES)
    changed = build({**local_config, "data": {**local_config["data"], "mix": {**local_config["data"]["mix"], "escape": False}}})
    result = pipeline.run(changed, STAGES)
    assert Path(result["directory"]) != Path(pipeline.Experiment(config).root)


def test_train_argv_mirrors_training_config(local_config: dict) -> None:
    config = build(local_config)
    assert config.training and config.model
    args = trainer.argv(config.training, config.model, Path("t"), Path("d"), Path("c"), Path("run"), Path("out"), 7)
    joined = " ".join(args)
    assert "--base-model Qwen/Qwen3.5-0.8B" in joined and "--eval-every 10" in joined and "--lr 5e-06" in joined
    assert "--patience" not in joined

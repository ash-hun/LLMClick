"""Stage orchestration: data -> benchmarks -> synthetic -> mix -> train -> evaluate, each skipped when its manifest entry matches."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.config.experiment import Experiment
from core.config.schema import STAGES, PipelineConfig
from core.modules.data import folds, mix, synthetic
from core.modules.evaluation import evaluator
from core.modules.tuning import tracker, trainer
from core.registry import BENCHMARKS, BUILDERS, TEACHERS
from core.utils.files import sha256_json
from jeff.data import validate, write_rows
from jeff.mix import read

logger = logging.getLogger(__name__)
Stage = Callable[[Experiment, PipelineConfig], dict[str, Any]]


def fingerprint(*parts: Any) -> str:
    return sha256_json(list(parts))


def stage_data(exp: Experiment, config: PipelineConfig) -> dict[str, Any]:
    assert config.data is not None
    public = exp.data / "public"
    rows = []
    built: dict[str, str] = {}
    for index, builder in enumerate(config.data.builders):
        out = exp.data / "builders" / f"{index:02d}-{builder.name}"
        out.mkdir(parents=True, exist_ok=True)
        path = BUILDERS.get(builder.name)(builder.params, out, config.seed)
        built[out.name] = str(path)
        rows.extend(read(path))
    validate(rows)
    split = folds.carve(rows, config.data.folds.dev, config.data.folds.temperature, config.seed)
    outputs = {name: str(public / f"{name}.jsonl") for name in split}
    for name, fold_rows in split.items():
        write_rows(public / f"{name}.jsonl", fold_rows)
    return {**outputs, "builders": built, "rows": {name: len(fold_rows) for name, fold_rows in split.items()}}


def stage_benchmarks(exp: Experiment, config: PipelineConfig) -> dict[str, Any]:
    if config.evaluation is None:
        return {}
    outputs: dict[str, Any] = {}
    for benchmark in config.evaluation.benchmarks:
        key = benchmark.name if len(benchmark.params) == 0 else f"{benchmark.name}-{sha256_json(benchmark.params)[:8]}"
        out = exp.data / "benchmarks" / key
        out.mkdir(parents=True, exist_ok=True)
        outputs[key] = str(BENCHMARKS.get(benchmark.name)(benchmark.params, out, config.seed))
    return outputs


def stage_synthetic(exp: Experiment, config: PipelineConfig) -> dict[str, Any]:
    assert config.data is not None
    setting = config.data.synthetic
    if not setting.enabled:
        return {}
    data = run_stage(exp, "data", config)
    env = TEACHERS.get(setting.teacher.name)(setting.teacher.params)
    path = synthetic.generate(exp.data / "synthetic", Path(data["train"]), setting.slots, config.seed, setting.focus,
                              setting.concurrency, env)
    return {"synthetic": str(path)}


def stage_mix(exp: Experiment, config: PipelineConfig) -> dict[str, Any]:
    assert config.data is not None
    data = run_stage(exp, "data", config)
    protected = [Path(p) for p in run_stage(exp, "benchmarks", config).values()]
    generated = run_stage(exp, "synthetic", config).get("synthetic")
    report = mix.run(Path(data["train"]), Path(data["dev"]), Path(data["temperature"]), protected,
                     Path(generated) if generated else None, config.data.mix, config.seed, exp.data / "mix")
    return {"train": report["train_file"], "dev": str(exp.data / "mix" / "dev.jsonl"),
            "temperature": str(exp.data / "mix" / "calibration.jsonl"), "report": str(exp.data / "mix" / "report.json"),
            "sizes": report["sizes"], "leaks": report["leaks"]}


def stage_train(exp: Experiment, config: PipelineConfig) -> dict[str, Any]:
    if config.training is None:
        return {"checkpoint": config.checkpoint} if config.checkpoint else {}
    assert config.model is not None
    mixed = run_stage(exp, "mix", config)
    selected = trainer.train(config.training, config.model, Path(mixed["train"]), Path(mixed["dev"]),
                             Path(mixed["temperature"]), exp.runs, exp.checkpoints, config.seed, config.device)
    tracker.sync(config.tracker, exp.runs / trainer.RUN_NAME, exp.key, config.model_dump(mode="json"))
    return {"checkpoint": str(selected), "run": str(exp.runs / trainer.RUN_NAME)}


def stage_evaluate(exp: Experiment, config: PipelineConfig) -> dict[str, Any]:
    if config.evaluation is None:
        return {}
    checkpoint = run_stage(exp, "train", config).get("checkpoint")
    if not checkpoint:
        raise RuntimeError("Nothing to evaluate: no trained checkpoint and no `checkpoint` in the config")
    benchmarks = {name: Path(path) for name, path in run_stage(exp, "benchmarks", config).items()}
    results = evaluator.evaluate(Path(checkpoint), benchmarks, exp.eval, config.evaluation.batch_size, config.device)
    return {"results": str(exp.eval / "results.json"), "overall": {name: r["accuracy"] for name, r in results.items()}}


STAGE_FUNCTIONS: dict[str, Stage] = {"data": stage_data, "benchmarks": stage_benchmarks, "synthetic": stage_synthetic,
                                     "mix": stage_mix, "train": stage_train, "evaluate": stage_evaluate}
# What each stage's fingerprint depends on: its config sections plus the outputs of the stages it reads.
DEPENDENCIES: dict[str, tuple[str, ...]] = {"data": (), "benchmarks": (), "synthetic": ("data",),
                                            "mix": ("data", "benchmarks", "synthetic"), "train": ("mix",),
                                            "evaluate": ("train", "benchmarks")}
SECTIONS: dict[str, tuple[str, ...]] = {"data": ("seed", "data"), "benchmarks": ("seed", "evaluation"),
                                        "synthetic": ("seed", "data"), "mix": ("seed", "data"),
                                        "train": ("seed", "model", "training", "device", "checkpoint"),
                                        "evaluate": ("evaluation", "device", "checkpoint")}


def stage_fingerprint(exp: Experiment, stage: str, config: PipelineConfig) -> str:
    upstream = [exp.manifest()["stages"].get(dep, {}).get("fingerprint") for dep in DEPENDENCIES[stage]]
    return fingerprint(stage, [config.section(s) for s in SECTIONS[stage]], upstream)


def run_stage(exp: Experiment, stage: str, config: PipelineConfig) -> dict[str, Any]:
    """Outputs of a stage: the manifest's if it is up to date, otherwise run it. Upstream stages settle first so the
    fingerprint sees their final state; running them later would make the second run recompute this stage."""
    for dependency in DEPENDENCIES[stage]:
        run_stage(exp, dependency, config)
    key = stage_fingerprint(exp, stage, config)
    cached = exp.stage_outputs(stage, key)
    if cached is not None:
        logger.info("[%s] %s: up to date", exp.key, stage)
        return cached
    logger.info("[%s] %s: running", exp.key, stage)
    outputs = STAGE_FUNCTIONS[stage](exp, config)
    exp.mark_done(stage, key, outputs)
    return outputs


def run(config: PipelineConfig, stages: list[str] | None = None) -> dict[str, Any]:
    wanted = list(STAGES) if stages is None else stages
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; choose from {STAGES}")
    exp = Experiment(config)
    exp.ensure()
    results = {stage: run_stage(exp, stage, config) for stage in STAGES if stage in wanted}
    return {"experiment": exp.key, "directory": str(exp.root), "stages": results}

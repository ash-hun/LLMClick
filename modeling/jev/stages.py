"""jev's stages: data -> benchmarks -> synthetic -> mix -> train -> validate -> evaluate."""

from pathlib import Path
from typing import Any, ClassVar

from jeff.data import validate, write_rows
from jeff.mix import read

from modeling.jev.registry import BENCHMARKS, BUILDERS, TEACHERS
from modeling.jev.tuning import history, trainer
from modeling.stages import TrainStage, ValidateStage
from modeling.jev.data import folds, mix, synthetic
from modeling.jev.evaluation import evaluator
from modeling.config import keyed_identity
from modeling.jev.config import JevConfig
from core.utils.files import sha256_json
from core.stage import Outputs, Stage
from modeling.tracker import History

SCALARS = ("count", "accuracy", "ece", "brier", "nll")  # the metrics validation bounds can name


class DataStage(Stage[JevConfig]):
    """Run every builder, concatenate, and carve the held-out folds by family."""
    name: ClassVar[str] = "data"

    def identity(self) -> Any:
        data = self.config.data
        return None if data is None else [self.config.seed, keyed_identity(data.builders), data.folds.model_dump()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        data = self.config.data
        if data is None:
            return {}
        rows = []
        built: dict[str, str] = {}
        for index, builder in enumerate(data.builders):
            self.progress.update(index, len(data.builders), builder.name)
            out = workdir / "builders" / f"{index:02d}-{builder.name}"
            out.mkdir(parents=True, exist_ok=True)
            path = BUILDERS.get(builder.name)(builder.params, out, self.config.seed)
            built[out.name] = str(path)
            rows.extend(read(path))
        self.progress.update(len(data.builders), len(data.builders), "folds")
        validate(rows)
        split = folds.carve(rows, data.folds.model_dump(), self.config.seed)
        outputs = {name: str(workdir / "public" / f"{name}.jsonl") for name in split}
        for name, fold_rows in split.items():
            write_rows(Path(outputs[name]), fold_rows)
        return {**outputs, "builders": built, "rows": {name: len(fold_rows) for name, fold_rows in split.items()}}


class BenchmarksStage(Stage[JevConfig]):
    """Freeze every evaluation set; the mix stage keeps training rows away from them."""
    name: ClassVar[str] = "benchmarks"

    def identity(self) -> Any:
        evaluation = self.config.evaluation
        return None if evaluation is None else [self.config.seed, keyed_identity(evaluation.benchmarks)]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        if self.config.evaluation is None:
            return {}
        benchmarks = self.config.evaluation.benchmarks
        outputs: Outputs = {}
        for index, benchmark in enumerate(benchmarks):
            self.progress.update(index, len(benchmarks), benchmark.name)
            key = benchmark.name if not benchmark.params else f"{benchmark.name}-{sha256_json(benchmark.params)[:8]}"
            out = workdir / key
            out.mkdir(parents=True, exist_ok=True)
            outputs[key] = str(BENCHMARKS.get(benchmark.name)(benchmark.params, out, self.config.seed))
        self.progress.update(len(benchmarks), len(benchmarks))
        return outputs


class SyntheticStage(Stage[JevConfig]):
    """Teacher-written questions with verify/review/repair; empty unless `data.synthetic.enabled`."""
    name: ClassVar[str] = "synthetic"
    requires: ClassVar[tuple[str, ...]] = ("data",)

    def identity(self) -> Any:
        data = self.config.data
        return None if data is None else [self.config.seed, data.synthetic.model_dump(mode="json")]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        data = self.config.data
        if data is None or not data.synthetic.enabled:
            return {}
        setting = data.synthetic
        env = TEACHERS.get(setting.teacher.name)(setting.teacher.params)
        path = synthetic.generate(workdir, Path(inputs["data"]["train"]), setting.slots, self.config.seed,
                                  setting.focus, setting.concurrency, env)
        return {"synthetic": str(path)}


class MixStage(Stage[JevConfig]):
    """Leak filter against the benchmarks and the validation fold, layouts, escape options, hijack attempts, size caps."""
    name: ClassVar[str] = "mix"
    requires: ClassVar[tuple[str, ...]] = ("data", "benchmarks", "synthetic")

    def identity(self) -> Any:
        data = self.config.data
        return None if data is None else [self.config.seed, data.mix.model_dump()]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        if self.config.data is None:
            return {}
        data = inputs["data"]
        protected = [Path(path) for path in inputs["benchmarks"].values()] + [Path(data["validation"])]
        generated = inputs["synthetic"].get("synthetic")
        report = mix.run(Path(data["train"]), Path(data["dev"]), Path(data["temperature"]), protected,
                         Path(generated) if generated else None, self.config.data.mix, self.config.seed, workdir)
        return {"train": report["train_file"], "dev": str(workdir / "dev.jsonl"),
                "temperature": str(workdir / "calibration.jsonl"), "report": str(workdir / "report.json"),
                "sizes": report["sizes"], "leaks": report["leaks"]}


class JevTrainStage(TrainStage[JevConfig]):
    """Full-weight SFT with checkpoint selection on dev NLL; without `training`, hands on the config's checkpoint."""
    requires: ClassVar[tuple[str, ...]] = ("mix",)
    sections: ClassVar[tuple[str, ...]] = ("seed", "model", "training", "device", "checkpoint")

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config = self.config
        if config.training is None:
            return {"checkpoint": config.checkpoint} if config.checkpoint else {}
        assert config.model is not None
        mixed = inputs["mix"]
        selected = trainer.train(config.training, config.model, Path(mixed["train"]), Path(mixed["dev"]),
                                 Path(mixed["temperature"]), workdir / "runs", workdir / "checkpoints", config.seed,
                                 config.device, self.progress)
        return {"checkpoint": str(selected), "run": str(workdir / "runs" / trainer.RUN_NAME)}

    def history(self, outputs: Outputs) -> History:
        return history.flatten(Path(outputs["run"])) if "run" in outputs else []


class JevValidateStage(ValidateStage[JevConfig]):
    """Accuracy and calibration on the validation fold, which neither training nor checkpoint selection saw."""
    requires: ClassVar[tuple[str, ...]] = ("train", "data")
    sections: ClassVar[tuple[str, ...]] = ("validation", "device")

    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        checkpoint = Path(inputs["train"]["checkpoint"])
        fold = inputs["data"].get("validation")
        if fold is None:  # an existing checkpoint with no data section: there is no fold, only the artifact to check
            if not checkpoint.exists():
                raise FileNotFoundError(f"checkpoint {checkpoint} does not exist")
            return {}
        overall = evaluator.evaluate(checkpoint, {"validation": Path(fold)}, workdir, self.config.validation.batch_size,
                                     self.config.device, self.progress)["validation"]
        return {name: float(overall[name]) for name in SCALARS}


class EvaluateStage(Stage[JevConfig]):
    """Accuracy, ECE, Brier and NLL of the validated checkpoint on every frozen benchmark."""
    name: ClassVar[str] = "evaluate"
    requires: ClassVar[tuple[str, ...]] = ("validate", "benchmarks")

    def identity(self) -> Any:
        evaluation = self.config.evaluation
        return None if evaluation is None else [evaluation.batch_size, self.config.device]

    def run(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        if self.config.evaluation is None:
            return {}
        benchmarks = {name: Path(path) for name, path in inputs["benchmarks"].items()}
        results = evaluator.evaluate(Path(inputs["validate"]["checkpoint"]), benchmarks, workdir,
                                     self.config.evaluation.batch_size, self.config.device, self.progress)
        return {"results": str(workdir / "results.json"),
                "overall": {name: result["accuracy"] for name, result in results.items()}}

"""Stages of `llm_merge`: the `train` stage merges safetensors files tensor by tensor, validation compares the loss to the base."""

import json
import shutil
import hashlib
from pathlib import Path
from typing import Any, ClassVar
from contextlib import ExitStack

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modeling.llm.models.llm_backbone import GGUF_FILE, HEAD_CONFIG, LLMBackbone, local_snapshot
from core.file_store import read_json, read_jsonl, staged_directory, write_json
from modeling.merge.merge_rule import TASK_RULES, Tensors, merge_tensor
from modeling.tuning.tuning_stages import MeasureStage, TuneStage
from modeling.tuning.training_loop import CHECKPOINT, SUMMARY
from modeling.tuning.training_method import fitting
from modeling.merge.merge_config import MergeConfig
from modeling.llm.methods.sft_method import SFT
from core.device_selector import resolve_device
from core.progress_reporter import Progress
from core.stage_base import Outputs

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
TIED = "lm_head.weight"  # the output layer of a model whose config ties it to the input embeddings


def weight_map(directory: Path) -> dict[str, str]:
    """Tensor name -> the safetensors file holding it. Quantized weights and decision heads are refused: averaging
    packed integers or heads with their own option codes gives nothing meaningful."""
    if (directory / GGUF_FILE).is_file() or "quantization_config" in config_of(directory):
        raise ValueError(f"{directory} holds quantized weights; merge the models before quantizing them")
    if (directory / HEAD_CONFIG).is_file():
        raise ValueError(f"{directory} has a decision head; llm_merge merges language models only")
    if (directory / INDEX).is_file():
        mapping: dict[str, str] = json.loads((directory / INDEX).read_text())["weight_map"]
    elif (directory / SINGLE).is_file():
        with safe_open(str(directory / SINGLE), "pt") as handle:
            mapping = {key: SINGLE for key in handle.keys()}
    else:
        raise ValueError(f"{directory} holds no safetensors weights")
    return {key: file for key, file in mapping.items() if not (key == TIED and tied(directory))}


def config_of(directory: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((directory / "config.json").read_text()) if (directory / "config.json").is_file() else {}
    return loaded


def tied(directory: Path) -> bool:
    """Whether the output layer is the input embeddings: some files repeat it (the Qwen3 release does), files that
    transformers saved leave it out, and loading restores it either way."""
    config = config_of(directory)
    return bool(config.get("tie_word_embeddings", (config.get("text_config") or {}).get("tie_word_embeddings", False)))


def seeded(seed: int, name: str) -> torch.Generator:
    """DARE's random drop for one tensor: fixed by the experiment's seed and the tensor's name, so a rerun or a
    resumed merge drops the same entries whatever order the tensors come in."""
    return torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:15], 16))


def merge_directories(config: MergeConfig, base: Path, models: list[Path], out: Path, progress: Progress) -> int:
    """Write the merged weights into `out` with the file layout of the base (task rules) or of the first model, and
    copy that model's config and tokenizer files. One tensor of every model is in memory at a time. The merged
    tensors take the first model's dtype: fine-tunes are saved in FP32, and their changes would round away in the
    base's bf16. Returns the number of merged tensors."""
    settings = config.merge
    task = settings.rule in TASK_RULES
    layout = base if task else models[0]
    maps = {directory: weight_map(directory) for directory in {layout, *models, *([base] if task else [])}}
    names = set(maps[models[0]])
    for directory in models:
        if set(maps[directory]) != names:
            differ = sorted(names.symmetric_difference(maps[directory]))[:5]
            raise ValueError(f"{directory} has other tensors than {models[0]} (e.g. {differ}); merged models must share one architecture")
    # a released base may hold tensors transformers does not load and fine-tunes therefore never save (Qwen3.5's
    # multi-token prediction layers); those stay out of the merged model, which loads the same way
    if task and not names <= set(maps[base]):
        raise ValueError(f"{base} lacks tensors the merged models have (e.g. {sorted(names - set(maps[base]))[:5]})")
    files = sorted({file for name, file in maps[layout].items() if name in names})
    weights = [model.weight for model in settings.models]
    count, dtype_of = 0, torch.float32
    with ExitStack() as stack:
        handles: dict[tuple[Path, str], Any] = {}

        def tensor(directory: Path, name: str) -> torch.Tensor:
            key = (directory, maps[directory][name])
            if key not in handles:
                handles[key] = stack.enter_context(safe_open(str(directory / key[1]), "pt"))
            loaded: torch.Tensor = handles[key].get_tensor(name)
            return loaded

        for index, file in enumerate(files):
            progress.update(index, len(files), f"merging {file}")
            merged: dict[str, torch.Tensor] = {}
            for name in sorted(key for key, owner in maps[layout].items() if owner == file and key in names):
                tensors = [tensor(directory, name) for directory in models]
                reference = tensor(base, name) if task else None
                if len({tuple(t.shape) for t in [*tensors, *([reference] if reference is not None else [])]}) > 1:
                    raise ValueError(f"tensor {name} has different shapes across the merged models")
                dtype_of = tensors[0].dtype
                merged[name] = merge_tensor(settings.rule, Tensors(tensors, weights, reference, settings.t, settings.density,
                                                                   seeded(config.seed, name))).contiguous()
            save_file(merged, str(out / file), metadata={"format": "pt"})
            count += len(merged)
        progress.update(len(files), len(files), "merged")
    for path in layout.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors"):
            shutil.copy2(path, out / path.name)
    if (out / INDEX).is_file():  # the index lists only the tensors written
        index = json.loads((out / INDEX).read_text())
        index["weight_map"] = {key: file for key, file in index["weight_map"].items() if key in names}
        index.get("metadata", {}).pop("total_size", None)  # it described the source files
        (out / INDEX).write_text(json.dumps(index, indent=2) + "\n")
    written = config_of(out)
    if "dtype" in written or "torch_dtype" in written:
        dtype = str(dtype_of).removeprefix("torch.")
        written.update({key: dtype for key in ("dtype", "torch_dtype") if key in written})
        (out / "config.json").write_text(json.dumps(written, indent=2) + "\n")
    return count


class MergeStage(TuneStage):
    """Measure the base model on the held-out rows, then merge. Named `train` so validation, evaluation recipes and
    `checkpoint: validate` read it like any trained model."""
    sections: ClassVar[tuple[str, ...]] = ("seed", "recipe", "model", "training", "method", "merge", "device")

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config = self.config
        assert isinstance(config, MergeConfig)
        checkpoint, summary = workdir / CHECKPOINT, workdir / SUMMARY
        if not (summary.exists() and checkpoint.is_dir()):
            shutil.rmtree(checkpoint, ignore_errors=True)  # a crash between writing the model and the summary
            for model in config.merge.models:
                if model.path is not None and not Path(model.path).is_dir():
                    raise FileNotFoundError(f"merged model {model.path!r} does not exist; run the experiment that writes it")
            backbone, method = config.build_backbone(), config.build_method()
            assert isinstance(backbone, LLMBackbone) and isinstance(method, SFT)
            self.progress.update(0, None, "loading base model")
            with backbone.loaded(resolve_device(config.device)):
                held_out = fitting(backbone, method, read_jsonl(Path(inputs["data"]["validation"])), config.training,
                                   self.progress, "validation")
                base_metrics = method.evaluate(backbone, held_out, config.validation.batch_size, self.progress)
            base = local_snapshot(*backbone.origin(None))
            models = [local_snapshot(*model.source()) for model in config.merge.models]
            with staged_directory(checkpoint) as staging:
                tensors = merge_directories(config, base, models, staging, self.progress)
            write_json(summary, {"rule": config.merge.rule, "models": len(models), "tensors": tensors,
                                 "base_loss": base_metrics["loss"], "base_perplexity": base_metrics["perplexity"]})
        return {"checkpoint": str(checkpoint), "run": str(workdir), **read_json(summary)}


class MergeMeasureStage(MeasureStage):
    """The merged model's held-out loss, and how far it is from the base model's on the same rows."""

    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        metrics = super().measure(workdir, inputs)
        base = float(inputs["train"]["base_loss"])
        return {**metrics, "base_loss": base, "loss_delta": metrics["loss"] - base}

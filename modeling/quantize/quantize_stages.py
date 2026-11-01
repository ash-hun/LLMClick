"""Stages of `llm_quantize`: the `train` stage quantizes instead of training, and validation compares the loss to the base."""

import shutil
from pathlib import Path
from typing import Any, ClassVar

from core.file_store import read_json, read_jsonl, staged_directory, write_json
from modeling.quantize.quantize_scheme import SCHEMES, QuantizeJob, gguf_version, size_mb
from modeling.tuning.tuning_stages import MeasureStage, TuneStage
from modeling.tuning.training_loop import CHECKPOINT, SUMMARY
from modeling.quantize.quantize_config import QuantizeConfig
from modeling.llm.models.llm_backbone import LLMBackbone
from modeling.tuning.training_method import fitting
from modeling.llm.methods.sft_method import SFT
from core.device_selector import resolve_device
from core.stage_base import Outputs


class QuantizeStage(TuneStage):
    """Measure the base model on the held-out rows, then let the scheme write the quantized model. Named `train` so
    validation, evaluation recipes and `checkpoint: validate` read it like any trained model."""
    sections: ClassVar[tuple[str, ...]] = ("seed", "recipe", "model", "training", "method", "quantization", "device")

    def identity(self) -> Any:
        """A GGUF file depends on the llama.cpp release that wrote it as well."""
        config = self.config
        assert isinstance(config, QuantizeConfig)
        return [super().identity(), gguf_version()] if config.quantization.name == "gguf" else super().identity()

    def train(self, workdir: Path, inputs: dict[str, Outputs]) -> Outputs:
        config = self.config
        assert isinstance(config, QuantizeConfig)
        checkpoint, summary = workdir / CHECKPOINT, workdir / SUMMARY
        if not (summary.exists() and checkpoint.is_dir()):
            shutil.rmtree(checkpoint, ignore_errors=True)  # a crash between writing the model and the summary
            backbone, method, device = config.build_backbone(), config.build_method(), resolve_device(config.device)
            assert isinstance(backbone, LLMBackbone) and isinstance(method, SFT)
            self.progress.update(0, None, "loading base model")
            with backbone.loaded(device):
                held_out = fitting(backbone, method, read_jsonl(Path(inputs["data"]["validation"])), config.training,
                                   self.progress, "validation")
                rows = fitting(backbone, method, read_jsonl(Path(inputs["data"]["train"])), config.training,
                               self.progress, "training")[: config.quantization.rows]
                base = method.evaluate(backbone, held_out, config.validation.batch_size, self.progress)
                calibration = [backbone.render(method.messages(row), False) for row in rows]
            origin, revision = backbone.origin(None)
            ignore = ["lm_head", *(f"re:.*{fragment}.*" for fragment in backbone.frozen)]
            self.progress.update(0, None, f"quantizing ({config.quantization.name})")
            with staged_directory(checkpoint) as staging:
                SCHEMES.get(config.quantization.name)(QuantizeJob(
                    loader=backbone.loader, origin=origin, revision=revision, tokenizer=backbone.tokenizer, device=device,
                    calibration=calibration, ignore=ignore, params=config.quantization.params, out=staging))
            write_json(summary, {"scheme": config.quantization.name, "base_loss": base["loss"],
                                 "base_perplexity": base["perplexity"], "size_mb": size_mb(checkpoint)})
        return {"checkpoint": str(checkpoint), "run": str(workdir), **read_json(summary)}


class QuantizeMeasureStage(MeasureStage):
    """The quantized model's held-out loss, and how far it moved from the base model's on the same rows."""

    def measure(self, workdir: Path, inputs: dict[str, Outputs]) -> dict[str, float]:
        metrics = super().measure(workdir, inputs)
        trained: dict[str, Any] = inputs["train"]
        return {**metrics, "base_loss": float(trained["base_loss"]), "loss_delta": metrics["loss"] - float(trained["base_loss"]),
                "size_mb": float(trained["size_mb"])}

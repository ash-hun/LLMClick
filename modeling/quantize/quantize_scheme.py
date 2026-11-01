"""Quantization schemes `llm_quantize` offers: a `quantization.name` resolves here to a function that writes the model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from modeling.llm.models.llm_backbone import quantization_config
from core.recipe_registry import Registry
from core.device_selector import Device

SCHEMES = Registry("quantization scheme")
MISSING_COMPRESSOR = "this scheme needs llm-compressor: `uv sync --extra quantize`"
# Weight-only schemes keep activations in 16 bits; W8A8 also quantizes them, so it needs the SmoothQuant pass first.
COMPRESSOR = {"awq": "W4A16_ASYM", "gptq": "W4A16", "w8a8": "W8A8", "fp8": "FP8_DYNAMIC"}
CALIBRATED = {"awq", "gptq", "w8a8"}
DAMPENING = 0.01


@dataclass(frozen=True)
class QuantizeJob:
    """Everything a scheme reads: where the source weights are, the calibration token ids, and where to write."""
    loader: Any                  # the transformers Auto class of the architecture
    origin: str
    revision: str | None
    tokenizer: Any
    device: Device
    calibration: list[list[int]]
    ignore: list[str]            # modules left in full precision: the output layer and any frozen tower
    params: dict[str, Any]
    out: Path


def cuda_only(job: QuantizeJob, name: str) -> None:
    if not job.device.startswith("cuda"):
        raise ValueError(f"quantization {name!r} needs a CUDA device, not {job.device!r}")


def bitsandbytes_scheme(job: QuantizeJob, name: str) -> None:
    """Load-time quantization serialized as is; no calibration rows are read."""
    cuda_only(job, name)
    model = job.loader.from_pretrained(job.origin, revision=job.revision, device_map={"": job.device},
                                       quantization_config=quantization_config(name.removeprefix("bnb_")),
                                       dtype=torch.bfloat16)
    model.save_pretrained(job.out)
    job.tokenizer.save_pretrained(job.out)


@SCHEMES.register("bnb_nf4")
def bnb_nf4(job: QuantizeJob) -> None:
    """4-bit NormalFloat with double quantization (bitsandbytes), the QLoRA format."""
    bitsandbytes_scheme(job, "bnb_nf4")


@SCHEMES.register("bnb_int8")
def bnb_int8(job: QuantizeJob) -> None:
    """LLM.int8() (bitsandbytes): 8-bit weights with outlier features kept in 16 bits."""
    bitsandbytes_scheme(job, "bnb_int8")


def modifiers(name: str, params: dict[str, Any], ignore: list[str]) -> list[Any]:
    """The llm-compressor recipe of a scheme; `params.scheme` overrides the preset, e.g. W8A16 for gptq. AWQ and
    SmoothQuant only rescale weights (llm-compressor's `transform` modifiers); a quantizing modifier follows them."""
    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError:
        raise ImportError(MISSING_COMPRESSOR) from None
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.gptq import GPTQModifier
    preset = str(params.get("scheme", COMPRESSOR[name]))
    # GPTQ's Hessian damping, a fraction of the diagonal mean; llm-compressor's default, raised per config when needed
    damping = float(params.get("dampening", DAMPENING))
    if name == "awq":
        return [AWQModifier(duo_scaling="both"), QuantizationModifier(targets=["Linear"], scheme=preset, ignore=ignore)]
    if name == "gptq":
        return [GPTQModifier(targets="Linear", scheme=preset, ignore=ignore, dampening_frac=damping)]
    if name == "w8a8":
        return [SmoothQuantModifier(smoothing_strength=float(params.get("smoothing", 0.8))),
                GPTQModifier(targets="Linear", scheme=preset, ignore=ignore, dampening_frac=damping)]
    return [QuantizationModifier(targets="Linear", scheme=preset, ignore=ignore)]


def calibration_dataset(calibration: list[list[int]]) -> Any:
    from datasets import Dataset
    return Dataset.from_dict({"input_ids": calibration, "attention_mask": [[1] * len(ids) for ids in calibration]})


def compressor_scheme(job: QuantizeJob, name: str) -> None:
    """One llm-compressor `oneshot` pass, saved compressed (compressed-tensors, which vLLM and transformers read)."""
    if name == "fp8":
        cuda_only(job, name)  # CPU builds of torch lack the float8 casts the scale search uses
    recipe = modifiers(name, job.params, job.ignore)
    from llmcompressor.entrypoints.oneshot import oneshot
    if name in CALIBRATED and not job.calibration:
        raise ValueError(f"quantization {name!r} needs calibration rows")
    model = job.loader.from_pretrained(job.origin, revision=job.revision, dtype=torch.bfloat16, device_map={"": job.device})
    if name in CALIBRATED:
        longest = max(len(ids) for ids in job.calibration)
        # unshuffled: the same rows in the same order give the same scales on a rerun
        try:
            oneshot(model=model, dataset=calibration_dataset(job.calibration), recipe=recipe, max_seq_length=longest,
                    num_calibration_samples=len(job.calibration), shuffle_calibration_samples=False)
        except RuntimeError as error:  # torch's LinAlgError is a RuntimeError that torch does not export to mypy
            if "positive-definite" not in str(error):
                raise
            # Qwen3-0.6B on the sample rows needs 0.1: some layers' Hessians are too ill-conditioned for 0.01
            damping = job.params.get("dampening", DAMPENING)
            raise ValueError(f"quantization {name!r}: a layer's Hessian is not positive-definite at dampening {damping}; "
                             "set quantization.dampening higher (e.g. 0.1) or calibrate on more varied rows") from error
    else:
        oneshot(model=model, recipe=recipe)
    model.save_pretrained(job.out, save_compressed=True)
    job.tokenizer.save_pretrained(job.out)


@SCHEMES.register("awq")
def awq(job: QuantizeJob) -> None:
    """Activation-aware 4-bit weights: the channels the calibration activations use most are scaled before rounding."""
    compressor_scheme(job, "awq")


@SCHEMES.register("gptq")
def gptq(job: QuantizeJob) -> None:
    """4-bit weights rounded layer by layer to cancel the error the calibration rows show."""
    compressor_scheme(job, "gptq")


@SCHEMES.register("w8a8")
def w8a8(job: QuantizeJob) -> None:
    """8-bit weights and activations: SmoothQuant moves activation outliers into the weights, then GPTQ."""
    compressor_scheme(job, "w8a8")


@SCHEMES.register("fp8")
def fp8(job: QuantizeJob) -> None:
    """FP8 weights with per-token dynamic activation scales; needs no calibration rows (Hopper or Ada GPUs run it)."""
    compressor_scheme(job, "fp8")


def size_mb(directory: Path) -> float:
    return sum(path.stat().st_size for path in directory.glob("*.safetensors")) / 1_000_000

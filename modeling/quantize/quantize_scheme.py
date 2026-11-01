"""Quantization schemes `llm_quantize` offers: a `quantization.name` resolves here to a function that writes the model."""

import os
import sys
import json
import subprocess
from typing import Any
from pathlib import Path
from dataclasses import dataclass

import torch

from modeling.llm.models.llm_backbone import GGUF_FILE, quantization_config
from core.recipe_registry import Registry
from core.device_selector import Device
from core.settings import get_settings

SCHEMES = Registry("quantization scheme")
MISSING_COMPRESSOR = "this scheme needs llm-compressor: `uv sync --extra quantize`"
# Weight-only schemes keep activations in 16 bits; W8A8 also quantizes them, so it needs the SmoothQuant pass first.
COMPRESSOR = {"awq": "W4A16_ASYM", "gptq": "W4A16", "w8a8": "W8A8", "fp8": "FP8_DYNAMIC"}
CALIBRATED = {"awq", "gptq", "w8a8"}
DAMPENING = 0.01
GGUF_TYPES = {"q8_0", "f16", "bf16"}  # what the Python converter writes itself; k-quants need llama.cpp's C++ tools
CONVERT_SECONDS = 4 * 3600


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


def llama_cpp() -> Path:
    directory = Path(get_settings().LLAMA_CPP_DIR)
    if not (directory / "convert_hf_to_gguf.py").is_file():
        raise FileNotFoundError(f"LLAMA_CPP_DIR {str(directory)!r} has no convert_hf_to_gguf.py; fetch it with environment/llama_cpp/setup.sh")
    return directory


def gguf_version() -> str | None:
    """The llama.cpp commit that writes GGUF files, so changing the pin rebuilds them; None before setup."""
    commit = Path(get_settings().LLAMA_CPP_DIR) / "COMMIT"
    return commit.read_text().strip() if commit.is_file() else None


def local_source(job: QuantizeJob) -> Path:
    """The converter reads a directory: a Hub model is taken from the Hugging Face cache, downloaded if missing."""
    if Path(job.origin).is_dir():
        return Path(job.origin)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:  # the base model was just loaded from the cache, so it is normally there; this also works offline
        return Path(snapshot_download(job.origin, revision=job.revision, local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(job.origin, revision=job.revision))


@SCHEMES.register("gguf")
def gguf(job: QuantizeJob) -> None:
    """A GGUF file for llama.cpp and Ollama (`type`: q8_0, the default, f16 or bf16), written by llama.cpp's converter
    in its own process. Only architectures transformers can read back from GGUF are accepted, so validation measures
    the file that was written."""
    from transformers.integrations.ggml import GGUF_CONFIG_MAPPING
    kind = str(job.params.get("type", "q8_0"))
    if kind not in GGUF_TYPES:
        raise ValueError(f"gguf type {kind!r}; one of {sorted(GGUF_TYPES)} (k-quants need llama.cpp's compiled tools)")
    source, directory = local_source(job), llama_cpp()
    family = json.loads((source / "config.json").read_text()).get("model_type")
    if family not in GGUF_CONFIG_MAPPING:
        raise ValueError(f"model_type {family!r} cannot be read back from GGUF by transformers, so it could not be validated")
    completed = subprocess.run([sys.executable, str(directory / "convert_hf_to_gguf.py"), str(source), "--outfile",
                                str(job.out / GGUF_FILE), "--outtype", kind], capture_output=True, text=True, timeout=CONVERT_SECONDS,
                               env={**os.environ, "PYTHONPATH": str(directory / "gguf-py")})
    if completed.returncode != 0:
        raise RuntimeError(f"llama.cpp's converter failed: {completed.stderr.strip().splitlines()[-1:]}")
    job.tokenizer.save_pretrained(job.out)


def size_mb(directory: Path) -> float:
    return sum(path.stat().st_size for pattern in ("*.safetensors", "*.gguf") for path in directory.glob(pattern)) / 1_000_000

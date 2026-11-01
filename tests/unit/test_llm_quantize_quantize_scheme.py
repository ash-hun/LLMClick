"""`llm_quantize` modeling/quantize/quantize_scheme.py: CUDA-only schemes, the llm-compressor recipes, the GGUF converter call."""

import json
import subprocess
from typing import Any
from collections.abc import Iterator
from pathlib import Path

import pytest

from modeling.quantize.quantize_scheme import QuantizeJob, bnb_nf4, fp8, gguf, modifiers, size_mb
from modeling.llm.models.llm_backbone import prequantized
from core.settings import get_settings


def job(device: str, tmp_path: Path) -> QuantizeJob:
    return QuantizeJob(loader=None, origin="x", revision=None, tokenizer=None, device=device, calibration=[],
                       ignore=["lm_head"], params={}, out=tmp_path)


def test_cuda_only_schemes_refuse_other_devices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'bnb_nf4' needs a CUDA device, not 'mps'"):
        bnb_nf4(job("mps", tmp_path))
    with pytest.raises(ValueError, match="'fp8' needs a CUDA device, not 'cpu'"):
        fp8(job("cpu", tmp_path))


def test_size_counts_only_weight_files(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"0" * 2_000_000)
    (tmp_path / "tokenizer.json").write_bytes(b"0" * 5_000_000)
    assert size_mb(tmp_path) == 2.0


def test_compressor_recipes() -> None:
    pytest.importorskip("llmcompressor")
    awq = modifiers("awq", {}, ["lm_head"])
    assert [type(m).__name__ for m in awq] == ["AWQModifier", "QuantizationModifier"] and awq[1].scheme == "W4A16_ASYM"
    gptq = modifiers("gptq", {"scheme": "W8A16", "dampening": 0.1}, ["lm_head"])[0]
    assert gptq.scheme == "W8A16" and gptq.dampening_frac == 0.1
    assert modifiers("w8a8", {}, [])[1].dampening_frac == 0.01
    assert [type(m).__name__ for m in modifiers("w8a8", {}, [])] == ["SmoothQuantModifier", "GPTQModifier"]
    assert modifiers("fp8", {}, ["lm_head"])[0].ignore == ["lm_head"]


class FakeTokenizer:
    def save_pretrained(self, directory: Path) -> None:
        (directory / "tokenizer.json").write_text("{}")


@pytest.fixture
def converter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[list[str]]]:
    """A llama.cpp directory with a placeholder converter, and subprocess.run recording the call instead of running it."""
    directory = tmp_path / "llama.cpp"
    directory.mkdir()
    (directory / "convert_hf_to_gguf.py").write_text("")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(directory))
    get_settings.cache_clear()
    calls: list[list[str]] = []

    def run(command: list[str], **keys: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        Path(command[command.index("--outfile") + 1]).write_bytes(b"GGUF")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    yield calls
    get_settings.cache_clear()


def gguf_job(tmp_path: Path, model_type: str, **params: Any) -> QuantizeJob:
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    (source / "config.json").write_text(json.dumps({"model_type": model_type}))
    return QuantizeJob(loader=None, origin=str(source), revision=None, tokenizer=FakeTokenizer(), device="cpu",
                       calibration=[], ignore=[], params=params, out=out)


def test_gguf_runs_the_converter_and_leaves_a_loadable_directory(tmp_path: Path, converter: list[list[str]]) -> None:
    job = gguf_job(tmp_path, "qwen3", type="bf16")
    gguf(job)
    command = converter[0]
    assert command[1].endswith("convert_hf_to_gguf.py") and command[2] == job.origin and command[-2:] == ["--outtype", "bf16"]
    assert prequantized(job.out) == {"gguf_file": "model.gguf"} and (job.out / "tokenizer.json").exists()
    assert size_mb(job.out) == pytest.approx(4 / 1_000_000)


def test_gguf_refuses_what_it_could_not_validate(tmp_path: Path, converter: list[list[str]]) -> None:
    with pytest.raises(ValueError, match="cannot be read back from GGUF"):
        gguf(gguf_job(tmp_path, "qwen3_5"))
    with pytest.raises(ValueError, match="k-quants"):
        gguf(gguf_job(tmp_path, "qwen3", type="q4_k_m"))
    assert converter == []


def test_gguf_without_llama_cpp_points_at_the_setup_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "missing"))
    get_settings.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="environment/llama_cpp/setup.sh"):
            gguf(gguf_job(tmp_path, "qwen3"))
    finally:
        get_settings.cache_clear()

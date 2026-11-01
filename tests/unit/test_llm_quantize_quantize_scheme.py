"""`llm_quantize` modeling/quantize/quantize_scheme.py: CUDA-only schemes refuse other devices, and the llm-compressor recipes."""

from pathlib import Path

import pytest

from modeling.quantize.quantize_scheme import QuantizeJob, bnb_nf4, fp8, modifiers, size_mb


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

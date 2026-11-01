"""`llm_sft` modeling/tuning/data_parallel.py: one device per process, from the card this job owns."""

import pytest

from modeling.tuning import data_parallel


def test_rank_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    assert data_parallel.rank_devices("cpu", 3) == ["cpu", "cpu", "cpu"]
    monkeypatch.setattr(data_parallel.torch.cuda, "device_count", lambda: 4)
    assert data_parallel.rank_devices("cuda", 2) == ["cuda:0", "cuda:1"]
    assert data_parallel.rank_devices("cuda:3", 2) == ["cuda:3", "cuda:0"]
    with pytest.raises(ValueError, match="4 CUDA devices are visible"):
        data_parallel.rank_devices("cuda", 5)
    with pytest.raises(ValueError, match="needs CUDA GPUs"):
        data_parallel.rank_devices("mps", 2)

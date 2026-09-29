import pytest
import torch

from jeff.models import device_from_environment


def test_device_setting_is_explicit_and_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEFF_DEVICE", raising=False)
    assert device_from_environment() is None  # unset: the model's own default
    monkeypatch.setenv("JEFF_DEVICE", "cpu")
    assert device_from_environment() == "cpu"
    monkeypatch.setenv("JEFF_DEVICE", "tpu")
    with pytest.raises(ValueError, match="use cuda, mps or cpu"):
        device_from_environment()
    monkeypatch.setenv("JEFF_DEVICE", "mps")
    if torch.backends.mps.is_available():
        assert device_from_environment() == "mps"
    else:
        with pytest.raises(RuntimeError, match="no mps device"):
            device_from_environment()

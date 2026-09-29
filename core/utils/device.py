"""Pick the accelerator once, explicitly: jeff's own default never chooses MPS, so Apple silicon would silently train on the CPU."""

from typing import Literal

Device = Literal["cuda", "mps", "cpu"]


def resolve_device(wanted: Device | None) -> Device:
    if wanted is not None:
        return wanted
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

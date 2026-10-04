"""Pick the accelerator once, explicitly: cuda, then mps, then cpu, unless the config names one."""

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

"""Pick the accelerator once, explicitly: cuda, then mps, then cpu, unless the config names one."""

import threading

Device = str  # "cuda", "cuda:1", "mps" or "cpu"
slot = threading.local()  # `slot.index`: which accelerator this worker thread owns, set by the job workers


def resolve_device(wanted: str | None) -> Device:
    """The config's device, or the best one present. A job worker that owns accelerator N gets "cuda:N", so
    several jobs on one multi-GPU server each train on their own card."""
    import torch
    device = wanted or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    index = getattr(slot, "index", None)
    if device == "cuda" and index is not None and torch.cuda.device_count() > 1:
        return f"cuda:{index % torch.cuda.device_count()}"
    return device


def seed(device: Device, value: int) -> None:
    """Seed the random generator of `device` only. `torch.manual_seed` would seed the CPU and every CUDA device at
    once, which is not this worker's business when another job trains on another card of the same server: it would
    reset that job's sampling mid-run. On a CPU run the CPU generator is the device's generator."""
    import torch
    if device.startswith("cuda"):
        with torch.cuda.device(device):
            torch.cuda.manual_seed(value)
    elif device == "mps":
        torch.mps.manual_seed(value)
    else:
        torch.manual_seed(value)

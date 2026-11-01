"""Data-parallel training: one process per device, gradients averaged between them, the first one reporting home."""

import queue
import socket
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from collections.abc import Callable

import torch

from core.progress_reporter import Progress
from core.device_selector import Device

POLL_SECONDS = 0.5
Step = Callable[[int, dict[str, float]], None]


@dataclass(frozen=True)
class World:
    rank: int
    size: int


def reduce_gradients(parameters: list[torch.nn.Parameter], world: World) -> None:
    """Average every trainable gradient over the processes; one that got no rows this step contributes zeros."""
    import torch.distributed as distributed
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        distributed.all_reduce(parameter.grad, op=distributed.ReduceOp.SUM)
        parameter.grad /= world.size


def reduce_mean(value: float, device: Device) -> float:
    import torch.distributed as distributed
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
    return float(tensor.item()) / distributed.get_world_size()


def rank_devices(device: Device, size: int) -> list[Device]:
    """One device per process: consecutive CUDA cards from the one this job owns, or the CPU (gloo; for tests)."""
    if device == "cpu":
        return ["cpu"] * size
    if not device.startswith("cuda"):
        raise ValueError(f"training.devices > 1 needs CUDA GPUs (or the CPU), not {device!r}")
    count, first = torch.cuda.device_count(), int(device.split(":")[1]) if ":" in device else 0
    if size > count:
        raise ValueError(f"training.devices={size} but {count} CUDA devices are visible")
    return [f"cuda:{(first + rank) % count}" for rank in range(size)]


class QueueProgress(Progress):
    """The first process's progress, sent to the parent, which owns the real reporter (terminal bar or job state)."""

    def __init__(self, channel: Any) -> None:
        self.channel = channel

    def update(self, done: int, total: int | None = None, note: str = "") -> None:
        self.channel.put(("progress", done, total, note))


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def worker(rank: int, size: int, port: int, devices: list[Device], config: Any, inputs: dict[str, Any], workdir: Path,
           channel: Any) -> None:
    """One process of the group: join it, train its share, leave it. Runs in a fresh interpreter (spawn)."""
    import torch.distributed as distributed
    from modeling.tuning.tuning_stages import tune
    device = devices[rank]
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    distributed.init_process_group("nccl" if device.startswith("cuda") else "gloo", init_method=f"tcp://127.0.0.1:{port}",
                                   rank=rank, world_size=size)
    try:
        primary = rank == 0
        progress = QueueProgress(channel) if primary else Progress()

        def on_step(step: int, metrics: dict[str, float]) -> None:
            channel.put(("step", step, metrics))

        tune(config, inputs, workdir, progress, on_step if primary else None, World(rank, size), device)
    finally:
        distributed.destroy_process_group()


def spawn(config: Any, inputs: dict[str, Any], workdir: Path, device: Device, progress: Progress, on_step: Step) -> None:
    """Run `config.training.devices` workers and relay the first one's progress and step metrics until all finish.
    A failure in any worker stops the rest and is raised here; so is a cancelled job, after the workers are stopped."""
    import torch.multiprocessing as multiprocessing
    from torch.multiprocessing.spawn import start_processes
    size = config.training.devices
    channel = multiprocessing.get_context("spawn").Queue()
    processes = start_processes(worker, args=(size, free_port(), rank_devices(device, size), config, inputs, workdir, channel),  # type: ignore[no-untyped-call]
                                nprocs=size, join=False, start_method="spawn")

    def relay() -> None:
        while True:
            try:
                message = channel.get_nowait()
            except queue.Empty:
                return
            if message[0] == "progress":
                progress.update(*message[1:])
            else:
                on_step(message[1], message[2])

    try:
        while not processes.join(timeout=POLL_SECONDS):
            relay()
        relay()
    except BaseException:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=10)
        raise

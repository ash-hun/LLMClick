"""Progress reporting: stages call `update`, the caller chooses how it is shown (terminal bars, job state, nothing)."""

from typing import Any

from tqdm import tqdm


class Cancelled(Exception):
    """Raised inside the pipeline when the job it runs for was cancelled; the next progress report is where it stops."""


class Progress:
    """Reports nothing; the base every reporter and every stage is written against."""

    def begin(self, experiment: str, stages: list[str]) -> None:
        pass

    def stage_started(self, stage: str) -> None:
        pass

    def update(self, done: int, total: int | None = None, note: str = "") -> None:
        """Progress inside the running stage; `total` may arrive late or never."""

    def stage_finished(self, stage: str, status: str) -> None:
        """`status` is done, cached or failed."""

    def end(self) -> None:
        pass


class BarProgress(Progress):
    """Two terminal bars: stages of the pipeline, and steps of the running stage. Silent when stderr is not a terminal."""

    def begin(self, experiment: str, stages: list[str]) -> None:
        self.outer = tqdm(total=len(stages), desc=experiment, unit="stage", position=0, disable=None)

    def stage_started(self, stage: str) -> None:
        self.outer.set_postfix_str(stage)
        self.inner = tqdm(desc=stage, position=1, leave=False, disable=None)

    def update(self, done: int, total: int | None = None, note: str = "") -> None:
        if self.inner.disable:
            return
        if total is not None:
            self.inner.total = total
        self.inner.n = done
        self.inner.set_postfix_str(note, refresh=False)
        self.inner.refresh()

    def stage_finished(self, stage: str, status: str) -> None:
        self.inner.close()
        self.outer.set_postfix_str(f"{stage}: {status}")
        self.outer.update(1)

    def end(self) -> None:
        self.outer.close()


class StateProgress(Progress):
    """Keeps the latest state so a job can be polled, and carries a cancellation request into the pipeline: a thread
    cannot be stopped from outside, so the pipeline stops itself at its next progress report after `cancel()`.
    A phase that reports nothing (a model download, say) is only interrupted once it ends."""

    def __init__(self) -> None:
        self.state: dict[str, Any] = {"experiment": None, "stages": {}, "current": None, "done": 0, "total": None,
                                      "note": ""}
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled("the job was cancelled")

    def begin(self, experiment: str, stages: list[str]) -> None:
        self.state.update(experiment=experiment, stages={stage: "pending" for stage in stages})

    def stage_started(self, stage: str) -> None:
        self.check()
        self.state["stages"][stage] = "running"
        self.state.update(current=stage, done=0, total=None, note="")

    def update(self, done: int, total: int | None = None, note: str = "") -> None:
        self.check()
        self.state.update(done=done, note=note)
        if total is not None:
            self.state["total"] = total

    def stage_finished(self, stage: str, status: str) -> None:
        self.state["stages"][stage] = status

    def end(self) -> None:
        self.state["current"] = None

    def snapshot(self) -> dict[str, Any]:
        return {**self.state, "stages": dict(self.state["stages"])}

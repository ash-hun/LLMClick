"""Stand-ins for wandb and lm-evaluation-harness, so the recipes report and benchmark in tests without them."""

from typing import Any


class FakeWandb:
    """Stands in for the wandb module: records every run that is opened, each logged step, and each finish."""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.steps: list[int] = []
        self.reports: list[dict[str, Any]] = []
        self.finished = 0
        outer = self

        class Session:
            def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
                if step is None:
                    outer.reports.append(metrics)  # an evaluation report: numbers, then a table
                    return
                assert all(key.startswith(("train/", "eval/")) for key in metrics)
                outer.steps.append(step)

            def finish(self) -> None:
                outer.finished += 1

        self.session = Session

    def init(self, **keys: Any) -> Any:
        self.runs.append(keys)
        return self.session()

    class Table:
        def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
            self.columns, self.data = columns, data


class FakeHarness:
    """Stands in for lm_eval: records every call and answers with the shape `simple_evaluate` returns."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def simple_evaluate(self, **keys: Any) -> dict[str, Any]:
        self.calls.append(keys)
        (task,) = keys["tasks"]
        return {"results": {task: {"alias": task, "acc,none": 0.25 * len(self.calls), "acc_stderr,none": 0.05}},
                "n-samples": {task: {"original": 100, "effective": keys["limit"]}}, "n-shot": {task: keys["num_fewshot"] or 0},
                "versions": {task: 1}, "higher_is_better": {task: {"acc": True}},
                "samples": {task: [{"doc_id": 0, "acc": 1.0}]}, "config": {"model": keys["model"]}}

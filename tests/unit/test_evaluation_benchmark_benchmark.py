"""`evaluation_benchmark` evaluation/benchmark.py: presets resolve with limits, long requests get smaller batches, every task exists."""

import pytest

from evaluation.benchmark import PRESETS, Benchmarks, flat, length_groups


def test_presets_and_tasks_resolve_with_limits() -> None:
    tasks = Benchmarks(preset="korean", tasks=[{"name": "kmmlu", "num_fewshot": 0, "limit": 7}, {"name": "gsm8k"}], limit=3).resolved()
    assert [task.name for task in tasks] == ["haerae", "kobest", "click", "hrm8k", "kmmlu", "gsm8k"]
    assert {task.name: task.limit for task in tasks} == {"haerae": 3, "kobest": 3, "click": 3, "hrm8k": 3, "kmmlu": 7, "gsm8k": 3}
    assert next(task for task in tasks if task.name == "kmmlu").num_fewshot == 0  # replaced the preset entry
    with pytest.raises(ValueError, match="preset or at least one task"):
        Benchmarks()
    with pytest.raises(ValueError, match="choose from"):
        Benchmarks(preset="nope")
    assert flat({"acc,none": 0.5, "acc_stderr,none": 0.01, "alias": "x", "bleu,none": "N/A"}) == {"acc": 0.5, "acc_stderr": 0.01}


def test_long_requests_get_smaller_batches_under_the_logit_limit() -> None:
    groups = length_groups([10, 300, 50, 120, 2000], batch_size=8, rows=1000)
    assert groups == [(1, [4]), (3, [1]), (8, [3, 2, 0])]  # longest first; 1000 // 300 = 3, 1000 // 120 = 8
    assert all(size * max([10, 300, 50, 120, 2000][p] for p in positions) <= 1000 for size, positions in groups if size > 1)
    assert length_groups([5, 5], batch_size=2, rows=1000) == [(2, [0, 1])]  # short prompts keep the configured batch


def test_every_preset_task_exists_in_the_harness() -> None:
    from lm_eval.tasks import TaskManager

    names = set(TaskManager().all_tasks)
    missing = [item["name"] for tasks in PRESETS.values() for item in tasks if item["name"] not in names]
    assert missing == []

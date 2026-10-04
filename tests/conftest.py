"""Small Example JSONL fixtures so data, benchmarks and mix stages run without downloads or a GPU."""

import json
from pathlib import Path

import pytest

CRITERIA = {"billing": "Charges or refunds.", "technical": "Something is broken.", "account": "Login or profile."}
LABELS = list(CRITERIA)


def example(prefix: str, index: int) -> dict:
    label = LABELS[index % len(LABELS)]
    return {"id": f"{prefix}-{index:04d}", "suite": prefix, "family": f"{prefix}-fam-{index}",
            "state": f"Customer message {prefix} {index}: my {label} problem needs attention today, please route it.",
            "question": {"type": "choice", "instructions": "Route the request to one department.", "criteria": CRITERIA},
            "label": label, "target": label, "source": {"dataset": "fixture"}}


def write(path: Path, prefix: str, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(example(prefix, i)) + "\n" for i in range(count)))
    return path


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    write(tmp_path / "train.jsonl", "fix", 60)
    write(tmp_path / "test.jsonl", "bench", 12)
    return tmp_path


@pytest.fixture
def local_config(fixture_dir: Path) -> dict:
    return {
        "pipeline": {"recipe": "jev", "name": "local-test", "seed": 7, "output_dir": str(fixture_dir / "output")},
        "data": {"builders": [{"name": "local_jsonl", "path": str(fixture_dir / "train.jsonl")}],
                 "folds": {"dev": 6, "temperature": 4, "validation": 5},
                 "mix": {"size": 100, "sweep_size": 10, "panel_layout": False, "escape": True, "adversarial": True}},
        "model": {"backbone": "qwen3_5", "name": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
        "training": {"epochs": 1, "lr": 5e-6, "eval_every": 10},
        "evaluation": {"benchmarks": [{"name": "local_jsonl", "path": str(fixture_dir / "test.jsonl")}]},
    }

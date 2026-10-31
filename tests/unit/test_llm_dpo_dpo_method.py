"""`llm_dpo` modeling/llm/methods/dpo_method.py: the loss starts at log 2 and the reference margins are cached."""

import json
import math
from pathlib import Path

import pytest

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.tuning.tuning_config import TrainingConfig
from modeling.llm.methods.dpo_method import DPO
from core.progress_reporter import Progress


def test_dpo_starts_at_log_two_and_caches_the_reference(backbone: TransformerBackbone, tmp_path: Path) -> None:
    rows = [json.loads(line) for line in Path("samples/llm_dpo.jsonl").read_text().splitlines()][:6]
    method = DPO(DPO.Config(beta=0.5), TrainingConfig(batch_size=4, max_length=64))
    prepared = method.prepare(backbone, rows, tmp_path, Progress())
    assert float(method.loss(backbone, prepared).detach()) == pytest.approx(math.log(2), abs=1e-4)  # policy == reference
    (tmp_path / "reference_margins.json").write_text(json.dumps([0.0] * 6))
    assert [row["_reference"] for row in method.prepare(backbone, rows, tmp_path, Progress())] == [0.0] * 6  # read, not recomputed

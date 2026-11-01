"""`llm_kto` modeling/llm/methods/kto_method.py: the loss starts at one half, references are cached, held-out gap."""

import json
from pathlib import Path

import pytest

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.tuning.tuning_config import TrainingConfig
from modeling.llm.methods.kto_method import KTO
from core.progress_reporter import Progress

ROWS = [json.loads(line) for line in Path("samples/llm_kto.jsonl").read_text().splitlines()][:6]


def test_kto_starts_at_one_half_and_caches_the_reference(backbone: TransformerBackbone, tmp_path: Path) -> None:
    method = KTO(KTO.Config(desirable_weight=1.0, undesirable_weight=1.0), TrainingConfig(batch_size=4, max_length=64))
    prepared = method.prepare(backbone, ROWS, tmp_path, Progress())
    assert prepared[0]["_mismatch"] == ROWS[1]["completion"] and prepared[-1]["_mismatch"] == ROWS[0]["completion"]
    assert float(method.loss(backbone, prepared).detach()) == pytest.approx(0.5, abs=1e-4)  # policy == reference
    assert method.metrics["kl"] == pytest.approx(0.0, abs=1e-5)
    (tmp_path / "reference_logps.json").write_text(json.dumps([[0.0, 0.0]] * 6))
    assert [row["_reference"] for row in method.prepare(backbone, ROWS, tmp_path, Progress())] == [0.0] * 6


def test_kto_rows_and_held_out_metrics(backbone: TransformerBackbone) -> None:
    method = KTO(KTO.Config(), TrainingConfig(max_length=64))
    with pytest.raises(ValueError, match="label must be"):
        method.check({"prompt": "q", "completion": "a", "label": "yes"})
    with pytest.raises(ValueError, match="completion"):
        method.check({"prompt": "q", "completion": "", "label": True})
    both = method.evaluate(backbone, ROWS, 4, Progress())
    assert both["gap"] == pytest.approx(both["desirable_logp"] - both["undesirable_logp"])
    assert set(method.evaluate(backbone, [row for row in ROWS if row["label"]], 4, Progress())) == {"desirable_logp"}

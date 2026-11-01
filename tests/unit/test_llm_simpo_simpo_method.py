"""`llm_simpo` modeling/llm/methods/simpo_method.py: margins are per-token means, and gamma shifts the loss."""

import json
from pathlib import Path

import torch
import pytest

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.tuning.tuning_config import TrainingConfig
from modeling.llm.methods.simpo_method import SimPO


def test_simpo_margin_is_the_difference_of_mean_token_log_probabilities(backbone: TransformerBackbone) -> None:
    rows = [json.loads(line) for line in Path("samples/llm_dpo.jsonl").read_text().splitlines()][:4]
    method = SimPO(SimPO.Config(), TrainingConfig(max_length=64))
    chosen, chosen_count, rejected, rejected_count = method.answers(backbone, rows)
    expected = chosen / chosen_count - rejected / rejected_count
    assert torch.allclose(method.margins(backbone, rows), expected)


def test_simpo_loss_at_zero_margin_is_softplus_of_gamma(backbone: TransformerBackbone, monkeypatch: pytest.MonkeyPatch) -> None:
    method = SimPO(SimPO.Config(beta=2.0, gamma=1.5), TrainingConfig(max_length=64))
    monkeypatch.setattr(method, "margins", lambda backbone, rows: torch.zeros(3))
    assert float(method.loss(backbone, [{}] * 3)) == pytest.approx(float(torch.nn.functional.softplus(torch.tensor(1.5))))
    assert method.metrics["accuracy"] == 0.0

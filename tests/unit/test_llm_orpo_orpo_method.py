"""`llm_orpo` modeling/llm/methods/orpo_method.py: the log-odds of a mean log-probability, and the loss it builds."""

import json
import math
from pathlib import Path

import torch
import pytest

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.llm.methods.orpo_method import ORPO, log_odds
from modeling.tuning.tuning_config import TrainingConfig


def test_log_odds_matches_the_formula_and_stays_finite_near_certainty() -> None:
    p = 0.25
    assert float(log_odds(torch.tensor(math.log(p)))) == pytest.approx(math.log(p / (1 - p)))
    assert math.isfinite(float(log_odds(torch.tensor(0.0))))


def test_orpo_loss_is_chosen_nll_plus_the_weighted_odds_term(backbone: TransformerBackbone) -> None:
    rows = [json.loads(line) for line in Path("samples/llm_dpo.jsonl").read_text().splitlines()][:4]
    method = ORPO(ORPO.Config(lam=0.5), TrainingConfig(max_length=64))
    loss = float(method.loss(backbone, rows).detach())
    assert loss == pytest.approx(method.metrics["nll"] + 0.5 * method.metrics["odds_ratio"], rel=1e-5)
    margins = method.margins(backbone, rows).detach()
    assert method.metrics["odds_ratio"] == pytest.approx(float(-torch.nn.functional.logsigmoid(margins).mean()), rel=1e-5)

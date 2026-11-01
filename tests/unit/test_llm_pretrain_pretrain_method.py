"""`llm_pretrain` modeling/llm/methods/pretrain_method.py: packing loses no token, and every token is a target."""

from pathlib import Path

import pytest

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.llm.models.backbone_config import LLMBackboneConfig
from modeling.llm.methods.pretrain_method import Pretrain, packed
from modeling.tuning.tuning_config import TrainingConfig
from core.progress_reporter import Progress


def test_documents_are_packed_end_to_end_with_an_end_token_between(bpe_model: Path) -> None:
    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(bpe_model)))
    backbone.load("cpu")
    texts = ["언어 모델을 미세조정하면 응답 품질이 좋아집니다.", "체크포인트는 검증을 통과한 뒤에만 넘깁니다."]
    blocks = packed(backbone, texts, 8)
    stream = [token for block in blocks for token in block]
    end = backbone.tokenizer.eos_token_id
    assert all(len(block) == 8 for block in blocks[:-1]) and 0 < len(blocks[-1]) <= 8
    assert stream == [*backbone.token_ids(texts[0]), end, *backbone.token_ids(texts[1]), end]
    rows = Pretrain(Pretrain.Config(), TrainingConfig(max_length=8)).prepare(backbone, [{"text": t} for t in texts], Path("."), Progress())
    assert [row["ids"] for row in rows] == blocks


def test_every_token_after_the_first_is_scored(bpe_model: Path) -> None:
    backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=str(bpe_model)))
    backbone.load("cpu")
    method = Pretrain(Pretrain.Config(), TrainingConfig(max_length=16))
    rows = [{"text": "파이프라인은 같은 설정을 다시 실행해도 같은 결과를 냅니다."}]
    metrics = method.evaluate(backbone, rows, 4, Progress())
    blocks = packed(backbone, [rows[0]["text"]], 16)
    assert metrics["tokens"] == sum(len(block) - 1 for block in blocks)
    assert metrics["nll_per_char"] == pytest.approx(metrics["loss"] * metrics["tokens"] / len(rows[0]["text"]))
    with pytest.raises(ValueError, match="non-empty"):
        method.check({"text": "  "})

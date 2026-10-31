"""`llm_sft` modeling/tuning/training_method.py: rows are tokenized once across the length check, the loss and evaluation."""

from typing import Any

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.tuning.tuning_config import TrainingConfig
from tests.recipe_inputs import SFT_ROWS as ROWS
from modeling.llm.methods.sft_method import SFT
from core.progress_reporter import Progress


def test_rows_are_tokenized_once_across_the_length_check_the_loss_and_evaluation(backbone: TransformerBackbone) -> None:
    from modeling.tuning.training_method import fitting

    class Counting:
        def __init__(self, inner: Any) -> None:
            self.inner, self.calls = inner, 0

        def __call__(self, *args: Any, **keys: Any) -> Any:
            self.calls += 1
            return self.inner(*args, **keys)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

    counting = Counting(backbone.tokenizer)
    backbone.tokenizer = counting
    training = TrainingConfig(batch_size=4, max_length=64)
    method = SFT(SFT.Config(), training)
    rows = fitting(backbone, method, ROWS, training, Progress(), "training")
    after_check = counting.calls
    assert after_check == len(ROWS)  # the length check tokenizes every row once
    method.loss(backbone, rows[:4])
    method.evaluate(backbone, rows[:4], 4, Progress())
    assert counting.calls == after_check + 4  # per row only the prompt prefix is new; the full conversation is cached
    method.loss(backbone, rows[:4])
    assert counting.calls == after_check + 4  # the second epoch tokenizes nothing
    ids = backbone.token_ids("\n")
    ids.append(0)  # a caller may extend what it got without touching the cache
    assert backbone.token_ids("\n") != ids

"""`llm_sft` modeling/llm/methods/sft.py: the loss targets are exactly the assistant turns."""

from modeling.llm.models.transformer import TransformerBackbone
from modeling.tuning.config import TrainingConfig
from modeling.llm.methods.sft import SFT


def test_targets_are_exactly_the_assistant_turns(backbone: TransformerBackbone) -> None:
    messages = [{"role": "user", "content": "What is 2 plus 3 ?"}, {"role": "assistant", "content": "5"},
                {"role": "user", "content": "What is 4 plus 4 ?"}, {"role": "assistant", "content": "8"}]
    method = SFT(SFT.Config(), TrainingConfig())
    ids, targets = method.encode(backbone, messages)
    assert backbone.tokenizer.decode([i for i, flag in zip(ids, targets, strict=True) if flag]) == "5 <|im_end|> 8 <|im_end|>"
    ids, targets = method.encode(backbone, messages, last_only=True)
    assert backbone.tokenizer.decode([i for i, flag in zip(ids, targets, strict=True) if flag]) == "8 <|im_end|>"

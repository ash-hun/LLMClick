"""The `reward_model` reward: a checkpoint `llm_reward` trained scores each completion against its prompt."""

import threading
from pathlib import Path
from typing import Any

import torch

from modeling.llm.models.transformer_backbone import TransformerBackbone
from modeling.llm.models.backbone_config import LLMBackboneConfig
from modeling.llm.methods.llm_method import conversation
from modeling.llm.methods.grpo_reward import REWARDS
from modeling.tuning.training_method import Row
from core.device_selector import resolve_device

LOCK = threading.Lock()  # one model, many reward threads (`reward_workers`): one forward pass at a time
# ponytail: one reward model per process, replaced when another path is asked for; a long-lived job server keeps the
# last one in memory until then. Release it after each job if servers ever run several reward models side by side.
LOADED: dict[str, TransformerBackbone] = {}


def scorer(path: str, template: dict[str, Any]) -> TransformerBackbone:
    """The reward model at `path`, loaded on the best device on first use. Any LLM architecture loads the same way:
    as a sequence classifier, whose class its own config names."""
    key = f"{path}|{sorted(template.items())}"
    if key not in LOADED:
        for loaded in LOADED.values():
            loaded.release()
        LOADED.clear()
        backbone = TransformerBackbone(LLMBackboneConfig(architecture="transformer", name=path, template=template))
        backbone.scorer = True
        backbone.load(resolve_device(None))
        backbone.model.eval()
        LOADED[key] = backbone
    return LOADED[key]


@REWARDS.register("reward_model")
def reward_model(completion: str, row: Row, params: dict[str, Any]) -> float:
    """The score `path` (an `llm_reward` checkpoint) gives the completion as the answer to the row's `prompt`.
    Unbounded; GRPO compares completions within their group, so only the order and spread matter."""
    path = params.get("path")
    if not isinstance(path, str) or not Path(path, "config.json").is_file():
        raise ValueError(f"the reward_model reward needs `path`, an llm_reward checkpoint directory; got {path!r}")
    with LOCK, torch.no_grad():
        backbone = scorer(path, dict(params.get("template") or {}))
        sequence = backbone.render([*conversation(row["prompt"]), {"role": "assistant", "content": completion}], False)
        return float(backbone.sequence_scores([sequence])[0])

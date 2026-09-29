"""Backbone families jeff can train; the entry says which jeff architecture loads it and which options it takes."""

from core.config.schema import ModelConfig
from core.registry import BACKBONES

BACKBONES.register("qwen3_5")({"architecture": "qwen", "supports_prompt_layout": True, "suggested_lr": 5e-6,
                               "example": "Qwen/Qwen3.5-0.8B"})
BACKBONES.register("gemma4")({"architecture": "decoder", "supports_prompt_layout": False, "suggested_lr": 2e-5,
                              "example": "google/gemma-4-E2B-it"})
BACKBONES.register("modernbert")({"architecture": "encoder", "supports_prompt_layout": False, "suggested_lr": 5e-5,
                                  "example": "answerdotai/ModernBERT-large"})


def train_args(model: ModelConfig) -> list[str]:
    args = ["--base-model", model.name, "--revision", model.revision]
    if model.prompt_layout:
        args += ["--prompt-layout", model.prompt_layout]
    return args

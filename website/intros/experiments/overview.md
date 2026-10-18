---
title: Experiments
description: Experiments built with LLMClick, grouped by the type of model they train, with every result and failure recorded.
---

# Experiments

Each experiment page records one model built end to end with these recipes: the data, every training run including
the ones that failed, the evaluation, and what was learned. Pages are grouped by the type of model, so experiments on
the same architecture stack up and can be read side by side.

| Model type | What it is | Catalog entry | Experiments |
|---|---|---|---|
| [Decoder LLM](./decoder-llm/overview) | decoder-only transformer, full attention in every layer | `transformer`: Qwen3-0.6B, 1.7B | [Jarvis inner-thought router](./decoder-llm/jarvis-router) (Qwen3-0.6B, chosen) |
| [Hybrid LLM (Gated DeltaNet)](./hybrid-llm/overview) | Gated DeltaNet linear-attention layers mixed with full-attention layers | `hybrid`: Qwen3.5-0.8B, 2B | compared in the [router experiment](./decoder-llm/jarvis-router) (Qwen3.5-0.8B) |
| SSM | state-space layers only (Mamba) | not in the catalog yet | none |
| [Speech models](./speech/overview) | a VITS voice (Piper); a small head on frozen speech features (openWakeWord) | `tts_piper`, `wakeword` | [Jarvis voice agent](./speech/jarvis-agent) (wake word and voice) |

A new experiment goes under the type of the model it ships. A type without a page gets one with its first experiment:
an intro under `website/intros/experiments/<type>/`, listed in `scripts/sync.py` and `sidebars.js`.

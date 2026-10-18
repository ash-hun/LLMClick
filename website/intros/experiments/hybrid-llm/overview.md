---
title: Hybrid LLM (Gated DeltaNet)
description: Experiments on hybrid models that mix Gated DeltaNet linear attention with full attention (Qwen3.5).
---

# Hybrid LLM (Gated DeltaNet)

A hybrid model mixes Gated DeltaNet layers (linear attention with a gated, delta-rule state: memory and time grow
linearly with the input) with a few full-attention layers. The catalog entry is `hybrid` (Qwen3.5-0.8B, Qwen3.5-2B;
the vision tower stays frozen); every LLM and decision recipe accepts it, and LoRA adapts the linear-attention
projections too. On a Mac without the fused kernels, its linear-attention layers run a slower fallback.

| Experiment | Role of the hybrid model | Result |
|---|---|---|
| [Jarvis inner-thought router](../decoder-llm/jarvis-router) | the backbone compared with Qwen3-0.6B | 166 of 167 held-out, ECE 0.006, 0.30 s a request; not chosen, because its one boundary mistake was sure (0.997) and it was four times slower |

No experiment ships a hybrid model yet; the first one gets its page here.

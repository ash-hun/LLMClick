---
title: Decoder LLM
description: Experiments on decoder-only transformers (Qwen3).
---

# Decoder LLM

A decoder-only transformer reads its input with full attention in every layer. The catalog entry is `transformer`
(Qwen3-0.6B, Qwen3-1.7B); every LLM and decision recipe accepts it. At small sizes it is the fastest option on a Mac:
in the router experiment Qwen3-0.6B answered in 0.08 s against 0.30 s for the hybrid Qwen3.5-0.8B.

| Experiment | Recipe | Model | Result |
|---|---|---|---|
| [Jarvis inner-thought router](./jarvis-router) | `llm_decision_sft`, pointer head | Qwen3-0.6B | 166 of 167 held-out, ECE 0.016, 0.08 s a request |

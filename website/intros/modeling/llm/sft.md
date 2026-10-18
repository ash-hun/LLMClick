---
title: llm_sft
sidebar_label: llm_sft
description: Supervised fine-tuning on conversations; the loss is on every assistant turn.
---

# `llm_sft`: conversations

Supervised fine-tuning teaches a model to answer the way your conversations answer. Use it when you have dialogues,
one turn or many, whose assistant turns are the replies you want. The model learns every assistant turn and only
those: user and system turns are context.

## Rows

```json
{"messages": [{"role": "user", "content": "What is 9 plus 2?"}, {"role": "assistant", "content": "9 plus 2 is 11."}]}
```

A row may start with a `system` turn, holds any number of user and assistant turns, and ends with an assistant turn. The base model's chat template
formats them, so a Qwen3 model and a Qwen3.5 model see their own format.

## Config

<!-- include: configs/llm/sft.yaml -->

`configs/llm/sft_transformer.yaml` is the same recipe on Qwen3-0.6B, and `configs/llm/sft_lora.yaml` trains a LoRA
adapter instead of every weight.

## Use

```bash
uv run llmclick run configs/llm/sft.yaml
```

- **Gate**: `validation.max: {loss: ...}` on the held-out conversations; `perplexity` is reported too.
- **Output**: `output/sft-qwen3.5-0.8b-<hash>/validate/checkpoint`, a `save_pretrained` directory with its tokenizer.
- **Next**: score it on your rows with [`evaluation_custom`](../../evaluation/custom) and against the base model with
  [`evaluation_compare`](../../evaluation/compare).

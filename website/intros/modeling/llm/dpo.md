---
title: llm_dpo
sidebar_label: llm_dpo
description: Direct Preference Optimization on pairs of a better and a worse answer.
---

# `llm_dpo`: preference pairs

DPO teaches a model which of two answers to prefer, without a reward model and without sampling. Use it after SFT,
when you can say for a prompt which answer is better (human ratings, a judge, a verifier that rejected one of them).
The base model is the reference: its log-likelihood margins are computed once before training, and the policy is
pushed to prefer `chosen` over `rejected` more than the reference does.

## Rows

```json
{"prompt": "What is 9 plus 2? Reply with the number only.", "chosen": "11", "rejected": "9"}
```

`prompt` is a string or a list of chat messages.

## Config

<!-- include: configs/llm/dpo.yaml -->

## Use

```bash
uv run llmclick run configs/llm/dpo.yaml
```

- **Gate**: `accuracy`, the share of held-out pairs where `chosen` is more likely than `rejected`; `margin` is the
  mean log-likelihood gap.
- **Settings**: `method.beta` bounds how far the policy moves from the reference; smaller lets it move further.
  Start from an SFT checkpoint with `model.init: output/<sft experiment>/validate/checkpoint`.
- **Output**: `output/dpo-qwen3.5-0.8b-<hash>/validate/checkpoint`. The [Data channel](../../data/synthetic) makes
  pairs with `select.assemble: dpo` (a judge's score gap decides which answer is `chosen`).

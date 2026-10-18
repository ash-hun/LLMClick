---
title: llm_decision_cispo
sidebar_label: llm_decision_cispo
description: Stage 2 of a reasoning decision model; reasoning that leads the head to the right option is reinforced.
---

# `llm_decision_cispo`: reason, then decide

The second stage of a reasoning decision model (Jeeves). It starts from a stage-1 checkpoint (`model.init`), samples
`method.group_size` reasoning chains per question, rewards each by the probability the head then gives the right
option, and reinforces the chains that beat their group's mean (CISPO: the importance weight of each reasoning token
is capped). The head's loss after reasoning and without reasoning is added, so the one-pass answer keeps working.

Use it when reasoning measurably helps on your questions: compare `accuracy` with `think_accuracy` in stage 1's
validation, or run [`evaluation_decision`](../../evaluation/decision) with `think: both`.

## Rows

The same Jev records as stage 1.

## Config

<!-- include: configs/llm/decision_cispo.yaml -->

## Use

```bash
uv run llmclick run configs/llm/decision_pointer.yaml    # stage 1; prints the experiment key
uv run llmclick run configs/llm/decision_cispo.yaml      # model.init points at stage 1's validate/checkpoint
```

- **Gate**: `accuracy` (one pass) with `think_accuracy` (after reasoning) next to it, both calibrated.
- **Cost**: each step forwards `batch_size` x `group_size` sequences; `method.max_think` caps a chain, and
  `length_hinge` makes long chains lose reward.
- **Settings**: `method.iterations` and `method.clip` reuse a sampled group for several steps (see
  [`llm_grpo`](../llm/grpo)).

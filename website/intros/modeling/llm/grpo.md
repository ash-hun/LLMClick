---
title: llm_grpo
sidebar_label: llm_grpo
description: Group Relative Policy Optimization; completions are sampled, rewarded and reinforced against their group's mean.
---

# `llm_grpo`: rewards

GRPO improves a model on tasks whose answers can be checked: arithmetic, an exact label, valid JSON, a rule. For
every prompt the model samples a group of completions, a reward function scores each, and the completions above their
group's mean are reinforced. No reward model and no value network: the group is the baseline.

## Rows

```json
{"prompt": "What is 9 plus 2? Reply with the number only.", "answer": "11"}
```

`prompt` is a string or chat messages; the other keys are what the reward reads. Shipped rewards: `exact_match`,
`contains`, `last_number` (the last number in the completion against `answer`, for GSM8K-style problems).

## Config

<!-- include: configs/llm/grpo.yaml -->

`configs/llm/grpo_gsm8k.yaml` trains on GSM8K word problems from the Hub with the `last_number` reward.

## Use

```bash
uv run llmclick run configs/llm/grpo.yaml
```

- **Gate**: `reward`, the mean reward of the greedy completion on held-out prompts.
- **Cost**: sampling dominates. `method.group_size` completions per prompt, each up to `method.max_new_tokens`;
  `training.max_length` must leave room for them, and the length check counts them before the first step.
- **Output**: `output/grpo-qwen3.5-0.8b-<hash>/validate/checkpoint`.

## Reusing sampled groups, and writing a reward

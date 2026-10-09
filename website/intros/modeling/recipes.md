---
title: Training recipes
description: Which recipe to pick for which data, and the settings that matter.
---

# Training recipes

Pick the recipe by the data you have, not by the method's reputation:

- **Conversations** (`messages` with user and assistant turns): `llm_sft`. The loss is on every assistant turn.
- **Instruction, input, output records**: `llm_instruction`. The loss is on the output only; an optional system prompt
  goes in front.
- **Pairs of a better and a worse answer** to the same prompt: `llm_dpo`. No reward model, no sampling; the base model
  is the reference and its margins are computed once before training.
- **Prompts with a checkable answer** (a number, an exact string, a rule): `llm_grpo`. The model samples a group of
  completions per prompt, a reward function scores them, and the ones above their group's mean are reinforced.
- **Typed decisions** about a state (choice, yes or no, a score): the decision recipes, described on their own page.
- **Query and document pairs** for retrieval: `embedding_contrastive`.

Every LLM recipe accepts the same catalog of base models (`transformer` for Qwen3, `hybrid` for Qwen3.5) and the same
training block: epochs, learning rate, batch and accumulation, sequence length, a LoRA adapter, resume snapshots,
and an evaluation every N steps on the held-out rows. By default every weight trains in FP32; `training.adapter`
switches to LoRA, which cuts memory to little more than the weights and folds the adapter into a plain checkpoint
when it saves.

Two things the recipes do before spending any GPU time: they check every row against the recipe's format, and they
measure every row against `training.max_length` (counting what the method will add, like sampled completions), so a
row that does not fit stops the run at the start instead of hours in.

The sections below cover the two settings that need more than a sentence: reusing sampled groups in the
reinforcement recipes, and writing a reward.

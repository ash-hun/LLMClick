---
title: LLM recipes
description: The four recipes that tune a generative language model, the base models they accept, and the training block they share.
---

# LLM recipes

Four recipes tune a generative language model. They differ in what a row is and what the loss rewards, and share
everything else: the base models, the training block, LoRA, resume and the validation gate.

| Recipe | A row is | The loss | Validation metrics |
|---|---|---|---|
| [`llm_sft`](./sft) | a conversation (`messages`) | next-token loss on every assistant turn | `loss`, `perplexity` |
| [`llm_instruction`](./instruction) | `instruction`, optional `input`, `output` | next-token loss on the output only | `loss`, `perplexity` |
| [`llm_dpo`](./dpo) | `prompt`, `chosen`, `rejected` | prefer `chosen` more than the base model does | `accuracy`, `margin` |
| [`llm_grpo`](./grpo) | `prompt` and what the reward reads (`answer`) | reinforce sampled completions above their group's mean reward | `reward` |

## Base models

| `model.architecture` | Models | What it is |
|---|---|---|
| `transformer` | Qwen3-0.6B, Qwen3-1.7B | a decoder-only transformer: full attention in every layer |
| `hybrid` | Qwen3.5-0.8B, Qwen3.5-2B | Gated DeltaNet linear-attention layers mixed with full-attention layers; the vision tower stays frozen |

Pin `model.revision` to a commit so a run today and a run next month load the same weights. `model.init` starts
from a checkpoint of an earlier experiment instead (`<experiment>/validate/checkpoint`).

## The training block

```yaml
training:
  epochs: 1
  lr: 1.0e-5
  batch_size: 4
  accumulation: 2                       # optimizer step every 2 batches
  max_length: 512                       # every row is measured against it before the first step
  eval_every: 50                        # optional: held-out metrics during training
  adapter: {name: lora, r: 16, alpha: 32, dropout: 0.05}   # optional: omit to train every weight in FP32
```

With `adapter` only the LoRA weights train and the resume snapshot holds just them; the adapter is folded into the
base weights when the checkpoint is written, so a LoRA checkpoint loads like a fully trained one. Which layers are
adapted comes from the architecture (attention projections; for `hybrid` also the linear-attention projections)
unless `targets` names others.

## Run one

```bash
uv run llmclick validate configs/llm/sft.yaml     # experiment key and stage plan, nothing runs
uv run llmclick run configs/llm/sft.yaml          # data, train, validate
```

The shipped `configs/llm/` files are pilots: sample rows from `samples/` and `max_steps: 4`. Point `data.sources` at
your rows and remove `max_steps` for a real run. The checkpoint lands at `output/<name>-<hash>/validate/checkpoint`
and loads with `AutoModelForCausalLM.from_pretrained`.

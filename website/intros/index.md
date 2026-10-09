---
title: What LLMClick is
slug: /
description: One YAML describes one custom model. LLMClick trains it, checks it, measures it, and can make the data for it.
---

# What LLMClick is

LLMClick builds custom models from a config file. You write one YAML that names a base model, the rows to train on and a
few settings, run one command, and get a trained checkpoint that has passed a validation gate. The same YAML run twice
builds nothing the second time, and a run that crashed halfway continues where it stopped.

Three channels share one runner:

- **Modeling** turns a config into a trained, validated model. Recipes cover supervised fine-tuning, instruction
  tuning, preference tuning (DPO), reinforcement with rewards (GRPO), decision models that answer typed questions,
  and contrastive training of embedding models. Full fine-tuning or LoRA, on a GPU or on Apple silicon.
- **Evaluation** scores what Modeling built: on your own rows, on public benchmarks, against the base model, with the
  overlap between the evaluation items and the training rows reported next to every score.
- **Data** makes the rows: synthetic conversations, preferences and problems from a teacher model (an API or a local
  model), with the filtering and leak removal the published recipes use.

The mental model is small. A **recipe** is a pipeline of **stages**; each stage's output directory is named by a
fingerprint of its inputs, so stages are cached and shared between experiments; an **experiment** is one config and
links to the stage directories it used. Everything else (the CLI, the HTTP API with background jobs, the tracker)
drives that runner.

Who it is for: people who need a small, specific model (0.6B to 2B parameters) tuned on their data and want
reproducibility without writing training loops. If you already have a training script you like, LLMClick's value
is the fingerprinted pipeline, the validation gate and the evaluation channel around it.

The table below lists the recipes and the models each one accepts.

---
title: llm_decision_sft
sidebar_label: llm_decision_sft
description: Stage 1 of a decision model; a head scores the options, optionally after a reasoning chain.
---

# `llm_decision_sft`: decide in one pass

The first, and often the only, stage of a decision model. The backbone reads the state and the question, a head
scores every option, and cross-entropy over the options trains both. With `method.think_fraction: 0` the result is a
one-pass decision model: one forward pass a question, as a router on every request needs. Above 0, that share of
the questions gets a reasoning chain sampled from the base model and inserted as context, so the head also learns to
decide after reasoning (the start of [`llm_decision_cispo`](./cispo)).

## Rows

Jev records (`state`, `questions`), as on the [family page](./overview). `training.batch_size` counts questions,
however many a record holds.

## Config

<!-- include: configs/llm/decision_readout.yaml -->

Other shipped configs: `configs/llm/decision_pointer.yaml` (LoRA, pointer head, reasoning chains as context: stage 1
of a reasoning decision model) and `configs/llm/route.yaml` (the inner-thought router, trained on rows from
`configs/data/route.yaml`).

## Use

```bash
uv run llmclick run configs/llm/decision_readout.yaml
```

- **Head**: `readout` scores one-token option codes (A, B, ...); `pointer` compares the decide position with each
  option's end, so options need no codes and their meanings are read.
- **Gate**: calibrated `accuracy`, `nll` and `ece` on the held-out questions; `method.calibration` of the training
  records is kept aside to fit the temperature.
- **Output**: `output/<name>-<hash>/validate/checkpoint` with `head.safetensors` and `head.json` next to the weights.
  Measure it with [`evaluation_decision`](../../evaluation/decision): accuracy, calibration and latency.

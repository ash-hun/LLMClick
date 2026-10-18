---
title: Modeling overview
description: What the Modeling channel builds, the recipe for each kind of model, and the three steps every recipe runs.
---

# Modeling overview

The Modeling channel turns one YAML into one trained model that has passed a validation gate. It covers five kinds
of model, from a chat model to the detector that wakes a voice agent, and every one of them is built the same way:
read and check the rows, train, validate on rows training never saw. Only a model that passes is handed on.

![Modeling channel flow](/img/modeling-flow.svg)

## What it builds

| Kind of model | Recipe (`pipeline.recipe`) | Learns from | Base models (`model.architecture`) | Result |
|---|---|---|---|---|
| Chat and instruction LLM | [`llm_sft`](./llm/sft), [`llm_instruction`](./llm/instruction) | conversations; instruction, input, output records | `transformer` (Qwen3-0.6B, 1.7B), `hybrid` (Qwen3.5-0.8B, 2B) | a `save_pretrained` checkpoint |
| Preference-tuned LLM | [`llm_dpo`](./llm/dpo) | a better and a worse answer to the same prompt | same | same |
| Reward-tuned LLM | [`llm_grpo`](./llm/grpo) | prompts with a checkable answer | same | same |
| Decision model | [`llm_decision_sft`](./decision/sft), [`llm_decision_cispo`](./decision/cispo) | a state and typed questions about it | same, plus a head: `readout` or `pointer` | the checkpoint with a calibrated head |
| Embedding model | [`embedding_contrastive`](./embedding/contrastive) | query, positive and optional negative texts | `bi_encoder` (Qwen3-Embedding-0.6B) | the checkpoint, pooling included |
| Voice (TTS) | [`tts_piper`](./speech/tts-piper) | recordings and their texts | a Piper (VITS) checkpoint, `medium` or `high` | `voice.onnx` and fixed replies as WAV |
| Wake word detector | [`wakeword`](./speech/wakeword) | recordings of the name and of everything else | openWakeWord's frozen features with a small head | `<phrase>.onnx` for openWakeWord |

## Pick a recipe by the data you have

- **Conversations** (`messages` with user and assistant turns): `llm_sft`.
- **Instruction, input, output records**: `llm_instruction`.
- **Pairs of a better and a worse answer**: `llm_dpo`.
- **Prompts with a checkable answer** (a number, an exact string, a rule): `llm_grpo`.
- **A state and a question with fixed options** (route, classify, rate): `llm_decision_sft`, then
  `llm_decision_cispo` if reasoning before the decision should help.
- **Query and document pairs** for retrieval: `embedding_contrastive`.
- **Recordings of a voice**: `tts_piper`. **Recordings of a name**: `wakeword`.

No rows yet? The [Data channel](../data/overview) makes them, in the format each recipe reads.

## What every recipe shares

- **One YAML, one experiment.** `pipeline.recipe` picks the recipe, the other sections say what to build. The
  experiment directory is `output/<name>-<hash>/`; the same YAML run twice builds nothing the second time.
- **The validation gate.** `validate` follows `train` and cannot be left out. `validation.min` and `validation.max`
  name the recipe's metrics; a miss stops the run, and `<experiment>/validate/checkpoint` exists only for a model
  that passed. Later recipes (evaluation, a second training stage through `model.init`) read that link.
- **Checks before spending compute.** Every row is checked against the recipe's format and measured against
  `training.max_length` before the first step, so a bad row stops the run at the start, not hours in.
- **Resume after a crash.** Training keeps a snapshot and continues from it with the same batches and the same
  random draws, so an interrupted run ends with the weights an uninterrupted one would have.

The pages below go one level down: a family's page explains what its recipes share, and each recipe's page shows
its row format, the YAML the repository ships, and how to run it.

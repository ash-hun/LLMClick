---
title: Evaluation overview
description: What the Evaluation channel measures for each kind of model, how a measurement flows, and which recipe to run.
---

# Evaluation overview

Training tells you the loss went down. Evaluation tells you whether the model got better at what you care about,
and whether that number can be trusted. The Evaluation channel measures any checkpoint Modeling built, or a base
model from the Hub, on items it never trained on, and reports how much of the evaluation set leaked into training
next to every score.

![Evaluation channel flow](/img/evaluation-flow.svg)

## What it measures, by kind of model

| Model | Recipe | Measured on | Headline numbers |
|---|---|---|---|
| LLM, decision or embedding model | [`evaluation_custom`](./custom) | your rows, in the format the model trained on | the training recipe's own validation metrics |
| Generative LLM | [`evaluation_benchmark`](./benchmark) | [public benchmarks](./benchmark) through lm-evaluation-harness (English and Korean presets) | accuracy or exact match per benchmark, with its standard error |
| Several checkpoints and the base | [`evaluation_compare`](./compare) | the same benchmarks, rows, decision questions or MTEB tasks for every run | the difference to the base with a verdict, a polygon chart |
| Decision model | [`evaluation_decision`](./decision) | Jev rows or the JevBench tiers, without and with reasoning | accuracy, NLL, ECE, latency |
| Embedding model | [`evaluation_embedding`](./embedding) | MTEB tasks | each task's main score |
| Wake word detector | [`evaluation_wakeword`](./wakeword) | voices and hours of audio it never heard, clean and in noise | miss rate, false accepts per hour, the threshold sweep |
| TTS voice | `tts_piper`'s own `validate` | held-out texts read by the voice, transcribed by Whisper | `wer` against the recordings' `reference_wer` |

## Which one to run

1. Right after training: `evaluation_custom` on rows you care about, with `checkpoint: base` once for the baseline.
2. Before trusting a fine-tune: `evaluation_compare` of the base and the trained checkpoint on a benchmark preset,
   so a gain on your task and a loss in general knowledge show up together.
3. For decision, embedding and wake word models: their own recipe, because their metrics differ (calibration and
   latency; retrieval scores; false accepts per hour).

Everything is a recipe on the same runner as training: a score is cached per checkpoint content, task and settings,
so comparing three checkpoints scores each benchmark once per checkpoint, and a benchmark added later is the only one
that runs. Reports land in `report.json`, go to wandb with `tracker.enabled`, and are listed by the API.

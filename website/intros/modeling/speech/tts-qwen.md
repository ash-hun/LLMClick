---
title: tts_qwen
sidebar_label: tts_qwen
description: Clones a voice with Qwen3-TTS from one of its recordings and checks it by listening.
---

# `tts_qwen`: a cloned voice

Qwen3-TTS reads any text in a voice it has heard once, from one recording and its exact text. Nothing is trained:
the recipe picks that recording from the voice's recordings, the one a naturalness predictor rates highest, and puts
the clone through the same listening test as [`tts_piper`](./tts-piper), with one more metric, how much the clone
still sounds like the recordings.

On the Jarvis voice the clone sounds about as natural as recorded people (UTMOS 4.07 against 4.11; the fine-tuned
Piper voice scores 2.92) and is heard better, at the cost of a 4.2 GB model that needs Apple silicon. The comparison
is in the [Jarvis agent experiment](../../experiments/speech/jarvis-agent#b10-a-cloned-voice).

## Config

<!-- include: configs/tts/qwen_jarvis.yaml -->

## Use

```bash
environment/qwen/setup.sh                         # once: Qwen3-TTS's locked environment (Apple silicon)
uv run llmclick run configs/data/speech.yaml      # the recordings
uv run llmclick run configs/tts/qwen_jarvis.yaml  # replace <hash> first
```

The voice is `output/jarvis-qwen-<hash>/validate/checkpoint/` (`reference.wav` and `voice.json`), and each fixed reply
is `validate/phrases/<name>.wav`.

## How it works

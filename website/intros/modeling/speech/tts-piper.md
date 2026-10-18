---
title: tts_piper
sidebar_label: tts_piper
description: Fine-tunes a Piper voice from a Piper checkpoint and checks it by listening.
---

# `tts_piper`: a voice

A Piper voice is a small VITS model that reads phonemes and runs anywhere as one ONNX file. This recipe starts from
an existing Piper checkpoint, trains it on recordings with their texts, exports the result, and then has Whisper
listen to the new voice reading texts it never trained on. The voice passes when Whisper understands it about as well
as it understands the recordings themselves.

The training itself is Piper's: the recipe runs Piper's own trainer as a separate program, in an environment of its
own (Piper is GPL), and adds what the rest of this repository gives every model: fingerprinted stages, resuming
after a crash, and a validation gate before the voice is passed on.

## Rows

`audio` (a WAV file) and `text`, from `rows.jsonl` that `data_speech` writes or a Piper `metadata.csv`
(`<file>|<text>` lines) of your own recordings.

## Config

<!-- include: configs/tts/piper_jarvis.yaml -->

## Use

```bash
environment/piper/setup.sh                        # once: Piper's locked environment
uv run llmclick run configs/data/speech.yaml      # the training set
uv run llmclick run configs/tts/piper_jarvis.yaml # replace <hash> first
```

The voice is `output/jarvis-medium-ft-<hash>/validate/checkpoint/voice.onnx` (`piper -m voice.onnx`), and each fixed
reply is `validate/phrases/<name>.wav`.

## How it works

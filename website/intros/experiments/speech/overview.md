---
title: Speech models
description: Experiments on the audio models of a voice agent, the wake word detector and the voice.
---

# Speech models

Two model types serve a voice agent's audio, and both are built by the speech recipes:

| Model type | What it is | Recipe |
|---|---|---|
| Wake word detector | openWakeWord's frozen front end (melspectrogram, speech embedding every 80 ms) with a small DNN head over the last 2 s | `wakeword` |
| Voice (TTS) | VITS: a text encoder over espeak-ng phonemes, a flow and a vocoder, exported as one ONNX file (Piper) | `tts_piper` |

| Experiment | Models | Result |
|---|---|---|
| [Jarvis voice agent](./jarvis-agent) | wake word "Jarvis" (B7), the Jarvis voice with nine read-backs | detector: recall 1.000, false accepts 0.037 on its held-out split; voice: wer 0.072, read-backs heard exactly |

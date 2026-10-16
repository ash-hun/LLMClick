---
title: Text-to-speech
description: tts_piper fine-tunes a Piper voice from a Piper checkpoint and checks it by listening.
---

# Text-to-speech: `tts_piper`

A Piper voice is a small VITS model that reads phonemes and runs anywhere as one ONNX file. This recipe starts from
an existing Piper checkpoint, trains it on recordings with their texts, exports the result, and then has Whisper
listen to the new voice reading texts it never trained on. The voice passes when Whisper understands it about as well
as it understands the recordings themselves.

The training itself is Piper's: the recipe runs Piper's own trainer as a separate program, in an environment of its
own, and adds what the rest of this repository gives every model: fingerprinted stages, resuming after a crash, and a
validation gate before the voice is passed on.

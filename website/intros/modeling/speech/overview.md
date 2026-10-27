---
title: Speech recipes
description: The recipes a voice agent's audio needs, the voice that answers and the detector that wakes it.
---

# Speech recipes

A voice agent hears and speaks. Three recipes build the audio models around it, two of them alternatives for the voice; what it decides stays in the agent
(an LLM or a decision model from the recipes above).

| Recipe | Builds | Starts from | Trained by | Validation |
|---|---|---|---|---|
| [`tts_piper`](./tts-piper) | the voice that speaks the replies, and fixed replies as WAV files | a Piper (VITS) checkpoint | Piper's own trainer, in its own environment | Whisper listens: `wer` against the recordings' own `reference_wer`, `phrase_wer`; UTMOS and DNSMOS score how clean it sounds |
| [`tts_qwen`](./tts-qwen) | the same voice and replies, cloned from one recording; far more natural, far larger | a Qwen3-TTS Base model (MLX, Apple silicon) | nothing: the most natural recording is the voice | the same listening test, plus `speaker_similarity` |
| [`wakeword`](./wakeword) | the detector that fires when it hears the agent's name | openWakeWord's frozen speech features | a small head, in this process, on the CPU | `recall`, `false_accepts`, `scenario` (the name, then a command) |

All read recordings that [`data_speech`](../../data/speech) makes from seeds (a teacher writes lines, a Piper voice
reads them, Whisper checks them), or recordings of your own.

```
mic -> [wake word: always on, tiny] -> "Jarvis" -> [speech to text: the command] -> agent -> [TTS: the reply]
         wakeword                                                                              tts_qwen or tts_piper
```

## A voice agent from these recipes

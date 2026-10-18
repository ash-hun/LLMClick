---
title: Speech datasets
description: data_speech makes recordings with their texts, read by a Piper voice and checked by Whisper.
---

# Speech datasets: `data_speech`

`data_speech` makes the recordings the speech recipes train and are measured on: a teacher writes lines (or the seeds
are the lines), a Piper voice reads each one, Whisper checks what it heard, and the recordings whose transcript
matches their text are written out with a manifest that carries the voice's terms.

| Shipped config | Makes | Read by | For |
|---|---|---|---|
| `data/speech.yaml` | assistant lines written by a local model, plus the agent's fixed replies | one voice (`jgkawell/jarvis`) | `tts_piper` |
| `data/wake_phrase.yaml` | the name, "Jarvis.", as a statement, question and call | 300 speakers of LibriTTS-R | `wakeword` positives |
| `data/wake_negatives.yaml` | near misses, the commands said alone, ordinary lines | 40 speakers | `wakeword` negatives |
| `data/wake_nearmiss.yaml` | 20 near misses of the name | 120 more speakers | `wakeword` negatives |
| `data/wake_eval_phrase.yaml`, `wake_eval_negatives.yaml` | the name, near misses and commands | VCTK, a voice training never heard | `evaluation_wakeword` |

Five stages take seeds to a directory Piper can train on. The first two are those of `data_synthetic` (seeds, and the
texts made from them); then `synthesize`, `verify` and `select`. Every recording and every transcript is kept under
its natural key, so a run that stops halfway continues where it stopped, and a run with stricter checks reuses the
recordings it already has.

## Stages and settings

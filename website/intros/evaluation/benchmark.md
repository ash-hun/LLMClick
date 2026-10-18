---
title: Public benchmarks
description: Every public benchmark LLMClick runs, grouped by the kind of model it measures, and how to run each group.
---

# Public benchmarks

Public benchmarks place a model among others and catch regressions a fine-tune can cause (forgetting general
knowledge while learning your task). This page lists every public benchmark LLMClick runs, grouped by the kind of
model it measures, and then how to run each group.

## At a glance

| Model measured | Benchmarks built in | Recipe | Install |
|---|---|---|---|
| Generative LLM | presets `small_general` (8 English benchmarks) and `korean` (5 Korean benchmarks); any other lm-evaluation-harness task by name | `evaluation_benchmark` | `uv sync --extra eval` |
| Decision model | JevBench public tiers: easy (48), standard (72), hard (111) | `evaluation_decision` | none (downloaded once, pinned) |
| Embedding model | any MTEB task by name; the shipped config runs STSBenchmark, NFCorpus, Banking77Classification | `evaluation_embedding` | `uv sync --extra eval` |
| Wake word detector | LibriSpeech test-clean (5.4 h) as audio that never says the name; the name and near misses read by unseen voices (VCTK) | `evaluation_wakeword` | ffmpeg, to convert LibriSpeech to WAV |
| TTS voice | none public; Whisper transcribes the voice in `tts_piper`'s validation | (`tts_piper`) | |

## Generative LLMs: `evaluation_benchmark`

`small_general` is what the small-model cards report, plus a cheap regression panel. A fine-tune that forgets shows
up first in the last four.

| Benchmark | What it measures | Few-shot | Harness task |
|---|---|---|---|
| MMLU | knowledge across 57 subjects, multiple choice | 5 | `mmlu` |
| MMLU-Pro | harder knowledge and reasoning, ten options | 5 | `mmlu_pro` |
| GSM8K | grade-school math word problems | 8 | `gsm8k` |
| IFEval | following verifiable instructions (length, format, keywords), no judge | 0 | `ifeval` |
| HellaSwag | commonsense completion of everyday situations | 10 | `hellaswag` |
| ARC-Challenge | grade-school science questions that simple retrieval misses | 25 | `arc_challenge` |
| TruthfulQA (MC2) | avoiding common misconceptions | 0 | `truthfulqa_mc2` |
| WinoGrande | commonsense pronoun resolution | 5 | `winogrande` |

`korean` holds original Korean exams and culture, not translations.

| Benchmark | What it measures | Few-shot | Harness task |
|---|---|---|---|
| KMMLU | Korean expert-level exam knowledge, 45 subjects | 5 | `kmmlu` |
| HAE-RAE Bench | Korean vocabulary, history, culture and reading | 0 | `haerae` |
| KoBEST | Korean understanding: BoolQ, COPA, HellaSwag, SentiNeg, WiC | 5 | `kobest` |
| CLIcK | Korean cultural and linguistic knowledge | 0 | `click` |
| HRM8K | Korean math reasoning | 5 | `hrm8k` |

## Decision models: JevBench

JevBench (MIT) is the reference evaluation of the decision recipes' origins: realistic states with `choice`, yes or
no and `score` questions, in three public tiers. `evaluation_decision` downloads them once at a pinned revision and
reports per tier the accuracy against the chance level of its option counts, plus NLL, ECE and latency, without and
with reasoning. Not yet: the JevBench coherence suite and the Jeeves transfer panel. Details on
[Decision models](./decision).

## Embedding models: MTEB

Any MTEB task runs by name through the `mteb` package, with the experiment's own pooling and instructions. The
shipped config covers three task types: STSBenchmark (sentence similarity), NFCorpus (medical retrieval) and
Banking77Classification (intent classification). Details on [Embedding models](./embedding).

## Wake word detectors: hours of speech and unseen voices

Wake word detectors are compared on two things: misses on the name said by voices the detector never heard, and false
accepts per hour of audio that never says it. LibriSpeech test-clean (5.4 hours of read speech, CC BY 4.0) is the
ambient audio; a Piper voice trained on VCTK (109 speakers of British English accents) reads the name, near misses and
commands through `data_speech`, so no voice is shared with training. The targets come from openWakeWord (0.5 false
accepts an hour, under 5% misses) and Picovoice's benchmark (one an hour in ten). Details on
[Wake word detectors](./wakeword).

## Running them

```bash
uv sync --extra eval                                  # lm-evaluation-harness and mteb
uv run llmclick run configs/evaluation/benchmark.yaml # an LLM checkpoint on small_general plus KMMLU
uv run llmclick run configs/evaluation/decision.yaml  # a decision checkpoint on the JevBench tiers
uv run llmclick run configs/evaluation/embedding.yaml # an embedding checkpoint on three MTEB tasks
uv run llmclick run configs/evaluation/wakeword.yaml  # a wake word detector on LibriSpeech and unseen voices
```

Each config names a trained experiment (`model.experiment`, `model.checkpoint: validate`). To score a base model
before any training, write `model: {name: Qwen/Qwen3.5-0.8B, revision: <commit>}` instead (benchmark and embedding
recipes). To see what training changed, put the base and the trained checkpoint through
[`evaluation_compare`](./compare): the same benchmarks, and a verdict on every difference.

Start with a small `limit` to see the pipeline work, then remove it for real numbers. A score only compares with
another score taken under the same few-shot count, prompt format, decoding and harness version; the report records
all of them. Which benchmark tells you what, and why these were chosen, is in the [benchmark survey](./survey).

## `evaluation_benchmark` in detail

---
title: Data overview
description: What the Data channel makes, for which recipe, how a generation run flows, and which teachers it can use.
---

# Data overview

Most of the work in a good fine-tune is the data. The Data channel makes the rows a Modeling recipe trains on, in
the exact format that recipe reads, with a manifest that says what came from where, how much was dropped at each
step and what it cost. Today rows are made by a teacher model from seeds; converting open datasets is designed.

![Data channel flow](/img/data-flow.svg)

## What it makes, for which recipe

| Recipe | Makes | For | Shipped configs |
|---|---|---|---|
| [`data_synthetic`](./synthetic) | conversations, instruction rows, preference pairs, prompts with answers, decision questions | `llm_sft`, `llm_dpo`, `llm_grpo`, `llm_decision_sft` (`select.assemble`: `sft`, `dpo`, `grpo`, `decision`) | `data/synthetic.yaml` (Korean conversations), `data/route.yaml` (router decisions) |
| [`data_speech`](./speech) | recordings with their texts, checked by Whisper | `tts_piper`, `wakeword`, `evaluation_wakeword` | `data/speech.yaml`, `data/wake_*.yaml`, `data/wake_eval_*.yaml` |
| `data_open` (designed, not built) | rows converted from open datasets | every recipe | |

## Teachers

| `teacher.name` | Calls | Cost |
|---|---|---|
| `anthropic` | the Anthropic API (`ANTHROPIC_API_KEY`) | from a price table or `price: [in, out]` per million tokens |
| `openai` | any chat-completions server: OpenAI, vLLM, LM Studio, gateways (`base_url`); reasoning models with `reasoning_effort` | the same |
| `ollama` | a local Ollama server | free |
| `local` | a checkpoint Modeling trained, or a Hub model | free |

## What every data recipe guarantees

- **Every teacher call is cached** under a key made from the request, so a run that crashes or runs out of budget
  continues where it stopped, and the same config run twice costs nothing.
- **The budget is checked before each uncached call** (`budget.max_usd`), and the cache is the ledger.
- **Rows leave in the target recipe's format**: the last check is that recipe's own row check.
- **Duplicates and evaluation items are removed** before the rows are written, so a benchmark score is not inflated
  by its own items.

A Modeling config reads the result through `data.sources`:
`{name: local_jsonl, path: output/<data experiment>/select/rows.jsonl}`.

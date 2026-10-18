---
title: Synthetic rows
description: data_synthetic makes training rows from seeds through a teacher model; what it can make, then each stage.
---

# Synthetic rows: `data_synthetic`

`data_synthetic` makes training rows for the LLM and decision recipes from seeds, through a teacher model. It works
as a funnel: the early stages make many candidates cheaply, the later ones throw most of them away for a reason that
is recorded. The published recipes the design follows (Magpie, Tulu 3, OpenThoughts, Evol-Instruct, Nemotron) all
agree on that shape: generate more than you keep, and let verifiers and judges choose.

## What it can make

| `select.assemble` | Row | For | Typical verification |
|---|---|---|---|
| `sft` | a conversation: the prompt and the best answer | `llm_sft` | a judge with a rubric |
| `dpo` | the prompt, the best and the worst answer by judge score | `llm_dpo` | a judge; `margin` sets the minimum gap |
| `grpo` | the prompt and its verified answer | `llm_grpo` | a reward against the answer, majority agreement |
| `decision` | a state (earlier turns and the request) and one choice question | `llm_decision_sft` | the teacher's route equals the seed's option, majority agreement |

| Generator (`prompts.generator`) | Makes from a seed |
|---|---|
| `topic_questions` | questions about a topic, each rewritten to be more specific (Nemotron) |
| `persona_task` | one request a persona would make (Tulu 3) |
| `document_qa` | a question the passage answers, with its answer; the passage becomes context |
| `labelled_requests` | requests that belong to one option of a decision, with 0 to 2 earlier turns, in several languages |
| `passthrough` | the seed is already the instruction |

## Run one

```bash
uv run llmclick validate configs/data/synthetic.yaml   # experiment key and stage plan
uv run llmclick run configs/data/synthetic.yaml        # rows at output/<name>-<hash>/select/rows.jsonl
```

Before a large run, do a small one (two seeds, two answers) with the teacher you intend to use and read the cache
directories: they hold every request and answer, and they show quickly whether the teacher follows the generator's
format. A small teacher often does not; the filters will then reject most of its output, which is the pipeline
working as designed.

## Stages and settings

---
title: Comparing checkpoints
description: evaluation_compare puts several checkpoints and the base model through the same measurements.
---

# Comparing checkpoints: `evaluation_compare`

A single score means little; a difference against the base model, with its uncertainty, is what tells you whether
training helped. This recipe takes several runs (the base model is required), scores them on the same measurements,
and reports every difference with a verdict.

| What a run is scored on | Section | Works for |
|---|---|---|
| public benchmarks | `benchmarks` | generative LLMs |
| your rows, with the recipe's metrics | `rows` | LLM, decision and embedding models |
| decision questions, without and with reasoning | `decision` | decision models |
| MTEB tasks | `embedding` | embedding models |

| Verdict | Meaning |
|---|---|
| decided | the difference lies outside the 95% interval of the two measurements |
| not decided (`~`) | inside it |
| open (`?`) | no interval exists (the recipes' own metrics) |

Nothing is measured twice: each run's stages are the ones the single-model recipes use, so results already computed
are reused and adding a run scores only that run. The output is a table, a polygon chart over the axes you define,
and a Markdown summary you can paste into a report.

## Config and details

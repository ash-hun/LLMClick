---
title: Public benchmarks
description: evaluation_benchmark runs lm-evaluation-harness tasks with one cached stage per benchmark.
---

# Public benchmarks: `evaluation_benchmark`

Public benchmarks place a model among others and catch regressions a fine-tune can cause (forgetting general
knowledge while learning your task). LLMClick runs them through lm-evaluation-harness, the standard tool, and adds
what the harness does not: a stage per benchmark cached by the checkpoint's content and the exact settings, presets
for small models and for Korean, and the contamination measurement in the report.

Start with a small `limit` to see the pipeline work, then remove it for real numbers. Remember that a score only
compares with another score taken under the same few-shot count, prompt format, decoding and harness version; the
report records all of them.

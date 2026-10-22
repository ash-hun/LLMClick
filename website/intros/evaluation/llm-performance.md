---
title: LLM Performance Survey
description: Benchmark scores of small LLMs (up to 2B, the catalog's and their peers), published and measured, as polygon charts, one source per chart.
---

# LLM Performance Survey

Before fine-tuning, pick the base. This page collects the scores that model authors published for small LLMs, the
catalog's (Qwen3-0.6B, Qwen3-1.7B, Qwen3.5-0.8B, Qwen3.5-2B) and their peers, adds the scores LLMClick measured itself,
and draws them as polygon charts, so a model's strengths and gaps show at a glance.

## The models

| Model | Parameters | Architecture | In the catalog | Figures |
|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | decoder-only transformer | `transformer` | Qwen3 report, non-thinking and thinking |
| Qwen3-1.7B | 1.7B | decoder-only transformer | `transformer` | all four |
| Qwen3.5-0.8B | 0.8B | hybrid: Gated DeltaNet and full attention | `hybrid` | Qwen3.5 card, non-thinking and thinking |
| Qwen3.5-2B | 2B | hybrid: Gated DeltaNet and full attention | `hybrid` | Qwen3.5 card, non-thinking and thinking |
| Gemma-3-1B-IT | 1B | decoder-only transformer | no | Qwen3 report, non-thinking |
| Qwen2.5-1.5B-Instruct | 1.5B | decoder-only transformer | no | Qwen3 report, non-thinking |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | decoder-only transformer | no | Qwen3 report, thinking |
| SmolLM2-1.7B-Instruct | 1.7B | decoder-only transformer | no | measured here only |
| Qwen3-4B-2507 | 4B | decoder-only transformer | no | Qwen3.5 card, non-thinking (a size up, for reference) |

## How to read the charts

- **One chart, one source.** Each chart is one table of one publication, so its models were measured under the same
  prompts, decoding and harness. Scores from different charts do not compare, even for the same model and benchmark
  (Qwen3-1.7B's non-thinking MMMLU is 48.3 in the Qwen3 report and 46.7 in the Qwen3.5 card).
- **Published, except the measured chart.** The four source charts carry the authors' numbers. The measured chart
  is LLMClick's own run of `configs/evaluation/small_llms.yaml`; see its section.
- **Every axis runs from 0 at the centre to 100 at the outer ring.** Hard benchmarks (LiveCodeBench, AIME) sit near
  the centre for every model at this size; compare models along an axis, not axes with each other.
- **Thinking and non-thinking are separate charts.** The same model with and without reasoning is two models for
  this purpose.

## Qwen3 report: non-thinking, up to 2B

<!-- polygon: qwen3-report-nonthinking -->

Qwen3-1.7B leads on seven of the eight axes; Qwen2.5-1.5B-Instruct leads GPQA-Diamond (29.8 against 28.6). Qwen3-0.6B
scores at or above Gemma-3-1B-IT everywhere except Arena-Hard (6.5 against 17.8), the one axis judged by a model on
open-ended answers.

## Qwen3 report: thinking, up to 2B

<!-- polygon: qwen3-report-thinking -->

With reasoning, Qwen3-1.7B leads on every axis. DeepSeek-R1-Distill-Qwen-1.5B, distilled on math and code, beats
Qwen3-0.6B on MATH-500, AIME'24, GPQA-Diamond and LiveCodeBench but falls behind on instruction following (IFEval
39.9) and tool calling (BFCL 14.0). Thinking moves Qwen3-1.7B most on math and code: MATH-500 from 73.0 to 93.4 and
LiveCodeBench from 11.6 to 33.2, against the non-thinking chart above.

## Qwen3.5 card: non-thinking

<!-- polygon: qwen35-card-nonthinking -->

Qwen3.5-2B is above Qwen3-1.7B on every axis but IFEval (61.2 against 68.2). Qwen3.5-0.8B is below Qwen3-1.7B on
all six. Qwen3-4B-2507, a size up, shows how much room is left at 4B.

## Qwen3.5 card: thinking

<!-- polygon: qwen35-card-thinking -->

Qwen3.5-2B is above Qwen3-1.7B on all eight axes, by the most on instruction following (IFBench 41.3 against 26.7)
and long context (LongBench v2 38.7 against 26.5). Qwen3.5-0.8B gains from reasoning on knowledge (MMLU-Pro 29.7 to
42.3) but loses on IFEval (52.1 to 44.0), and its GPQA is 11.9. The card does not say which mode its Qwen3-1.7B
column used; its MMLU-Redux and IFEval equal the Qwen3 report's thinking numbers.

## Measured here: up to 2B

<!-- polygon: measured-small-llms -->

Measured with `configs/evaluation/small_llms.yaml`: every model instruction-tuned and pinned to a Hub commit, one
harness (lm-evaluation-harness 0.4.13), greedy decoding, no thinking. Multiple choice is scored by likelihood without
the chat template, GSM8K and IFEval through it. The config caps each task (200 items for ARC-C, HellaSwag, WinoGrande
and TruthfulQA, 100 for GSM8K and IFEval, 10 per subject for MMLU and KMMLU), so one standard error is about 2 points
on MMLU and KMMLU, 3 to 3.5 on the 200-item tasks and 5 on GSM8K and IFEval. Two models differ at 95% only when the
gap exceeds about 2.8 standard errors (1.96 times the root of two), the rule `evaluation_compare` reports by.

- Qwen3.5-2B and Qwen3-1.7B are first and second on ARC-C (55.5 and 53.5) and GSM8K (69.0 and 68.0), past the bar
  against the three models up to 1B on both. Qwen3.5-2B is also first on KMMLU (38.4), past the bar against those
  three. No gap between Qwen3.5-2B and Qwen3-1.7B passes it on any benchmark.
- Qwen2.5-1.5B-Instruct is uneven: highest on MMLU (63.2, level with Qwen3-1.7B's 62.8) but last on ARC-C (27.0, near
  the 25 a four-way guess gives, past the bar against every other model) and lowest on IFEval (41.0).
- SmolLM2-1.7B-Instruct is highest on commonsense (HellaSwag 66.5, WinoGrande 70.5), past the bar only against some
  models, and last on Korean: KMMLU 20.0, past the bar against every other model and below the four-way guess by more
  than two standard errors.
- Gemma-3-1B-IT is last on knowledge (MMLU 42.5), past the bar against every other model.
- TruthfulQA has no gap past the bar: all seven sit between 40.7 and 47.9.

The published charts above score some of the same models, under other prompts and harnesses, so their numbers do not
compare with these. Read side by side, the Qwen3.5 card's one exception, IFEval, where Qwen3-1.7B is above Qwen3.5-2B
(68.2 against 61.2), shows here too (67.0 against 62.0), inside the bar at 100 items. Llama-3.2-1B-Instruct is left
out because it needs its licence accepted on the Hub.

## Your own models on the same chart

`evaluation_compare` draws the same polygon (`axes.svg` in its report directory, by the same function) for the runs
you measure: the base model and your checkpoints, scored by one harness under one set of settings, which is the
comparison these published charts cannot give.

```yaml
pipeline: {recipe: evaluation_compare, name: base-vs-finetune}
runs:
  - {label: qwen3-0.6b, architecture: transformer, name: Qwen/Qwen3-0.6B, revision: c1899de289a04d12100db370d81485cdf75e47ca}
  - {label: finetune, experiment: output/<experiment>, checkpoint: validate}
benchmarks: {preset: small_general, limit: 100}
axes: {knowledge: [mmlu, mmlu_pro], math: [gsm8k], instructions: [ifeval], commonsense: [hellaswag, arc_challenge, winogrande]}
```

To add a published chart here, add a figure to `website/data/llm-performance.yaml` (one source's table, numbers as
printed, the source and its link) and a `<!-- polygon: <id> -->` line to this page's intro, then run `npm run sync`.

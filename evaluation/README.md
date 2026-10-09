# Evaluation channel

Status: milestones 1 and 2 built (`evaluation_custom`, `evaluation_benchmark`); the rest is designed below, after a
short survey of the benchmark landscape (October 2026). The survey's sources are at the end.

## Built: `evaluation_custom`

```yaml
pipeline: {recipe: evaluation_custom, name: sft-on-my-rows}
model:
  experiment: output/sft-qwen3.5-0.8b-<hash>   # the experiment's config.yaml says how the model loads and is measured
  checkpoint: validate                          # validate | train | base (what it started from) | a checkpoint directory
data:
  sources: [{name: local_jsonl, path: my-rows.jsonl}]   # rows in the experiment's recipe format
evaluation: {batch_size: 8}
```

Stages: `rows` (read and check against the recipe) -> `score` (load the checkpoint, the recipe's `evaluate`) ->
`report` (`report.json`: model, settings, rows, scores, contamination). `base` is the baseline: the weights the
experiment started from, measured the same way. Rows over the experiment's `training.max_length` follow its
`training.overflow`. The score stage's fingerprint is the experiment's config and the checkpoint's content, so a
rebuilt experiment is a new evaluation and an unchanged one is cached.

## Built: `evaluation_benchmark`

```yaml
pipeline: {recipe: evaluation_benchmark, name: sft-benchmarks}
model: {experiment: output/sft-qwen3.5-0.8b-<hash>, checkpoint: validate}
benchmarks:
  preset: small_general            # or korean; `tasks` are added to it, a task named twice takes the listed settings
  tasks: [{name: kmmlu, num_fewshot: 5}]
  limit: 100                       # smoke: 100; standard: 1000; full: remove
decoding: {temperature: 0, max_new_tokens: 1024}
chat_template: false               # true for instruction-tuned checkpoints
evaluation: {batch_size: 8}
```

Needs `uv sync --extra eval` (lm-evaluation-harness; not in the Docker image unless built with it). One stage per
benchmark, named `score:<task>`, whose fingerprint is the checkpoint's content, the task, its few-shot count and
limit, the decoding settings, the device and the harness version: a benchmark added to the list is the only one
scored, and two evaluations of the same checkpoint share every matching benchmark run. Each stage keeps the
harness's full `results.json` and every scored item in `samples.jsonl`; the report holds the flattened scores
(`acc`, `acc_stderr`, ...), the item count, the few-shot count and the limit per benchmark, plus the settings.

Presets: `small_general` (mmlu, mmlu_pro, gsm8k, ifeval, hellaswag, arc_challenge, truthfulqa_mc2, winogrande)
and `korean` (kmmlu, haerae, kobest, click, hrm8k). Any other lm-evaluation-harness task name works in `tasks`.
Not yet: thinking mode on or off for Qwen3 checkpoints (the harness is run with the chat template as it is), and
a Hub model without an experiment (use `checkpoint: base` of an experiment that starts from it).

## Purpose

Score trained models beyond the validation gate: on public benchmarks, on custom evaluation sets, and across
several checkpoints at once, with a report a person can read. The question it answers is "which checkpoint of
this model is best, on what, and compared with the base model".

## How it differs from `validate` and from `training.eval_every`

| | Rows | Checkpoints | Decides | Where |
|---|---|---|---|---|
| `training.eval_every` | the experiment's own held-out rows | the weights at that step | nothing; a curve next to the loss | `training.jsonl`, tracker |
| `validate` | the experiment's own held-out rows | the one it trained | pass or fail against `validation.min` / `max` | `validation.json` |
| Evaluation channel | any benchmark or evaluation set | any number, plus the base model | nothing; results kept for comparison | `output/<evaluation>/report.json`, a chart |

## Survey: what to measure for 0.6B to 2B fine-tuned models

**Assumptions.** Models are the catalog's (Qwen3-0.6B/1.7B, Qwen3.5-0.8B/2B, Qwen3-Embedding-0.6B) and their
fine-tunes; one GPU or a Mac; the comparison that matters is checkpoint against checkpoint and against the base
model, not against frontier models. Survey depth: light (direct searches, no literature review).

### Benchmarks by what they tell

| Axis | Benchmark | Why this one | Harness task |
|---|---|---|---|
| Knowledge | MMLU-Redux, MMLU-Pro | what the Qwen3.5 small-model cards report, so base-model numbers exist to compare with | `mmlu_redux`, `mmlu_pro` |
| Hard knowledge | GPQA Diamond | graduate-level, Google-proof; low for small models but moves with reasoning training | `gpqa` |
| Math | GSM8K, MATH-500 | saturated for frontier models, still discriminative for small and fine-tuned ones | `gsm8k`, `minerva_math`/`math500` |
| Instruction following | IFEval | verifiable constraints, no judge; reported for Qwen3.5-0.8B/2B (52.1 / 61.2 non-thinking) | `ifeval` |
| Commonsense, regression panel | HellaSwag, ARC-Challenge, TruthfulQA, Winogrande | cheap log-likelihood tasks; a fine-tune that forgets shows up here first | `hellaswag`, `arc_challenge`, `truthfulqa_mc2`, `winogrande` |
| Code | HumanEval+, MBPP+ (EvalPlus) | execution-based; only when the fine-tune targets code | `humaneval`, `mbpp` (EvalPlus variants) |
| Korean | KMMLU, KMMLU-Redux/Pro, HAE-RAE Bench, KoBEST, CLIcK, HRM8K | original Korean exams and culture, not translations; all present in lm-evaluation-harness | `kmmlu`, `haerae`, `kobest`, `click`, `hrm8k` |
| Long context | RULER, LongBench | only when the fine-tune changes context handling; expensive | `ruler`, `longbench` |
| Decision models | JevBench (easy, standard, hard tiers; 231 public items), plus the Jeeves transfer panel: MMLU, MMLU-Pro, PAWS, QNLI, SciQ, TweetEval offensive, Emotion | the reference evaluation of the decision recipes' origins; accuracy and ECE, with and without reasoning, plus latency | own runner (`DecisionMethod.evaluate` + JevBench format) |
| Embeddings | MTEB v2: MTEB(eng, v2) retrieval subset; Korean retrieval tasks from MMTEB | the standard; Qwen3-Embedding is reported on it | `mteb` package |

Base-model reference points (Qwen3.5 small-model cards, non-thinking unless noted):

| Benchmark | Qwen3.5-0.8B | Qwen3.5-2B |
|---|---|---|
| MMLU-Pro | 29.7 (thinking 42.3) | 55.3 (thinking 66.5) |
| MMLU-Redux | 48.5 (thinking 59.5) | 69.2 (thinking 79.6) |
| IFEval | 52.1 | 61.2 |
| SuperGPQA | 16.9 | 30.4 |
| MMMLU | 34.1 | 56.9 |

These numbers are the vendor's, under the vendor's prompts and decoding. The plan treats them as orientation only:
the channel measures the base model itself, under the same settings as the checkpoints, and that is the baseline.

### Harness

- **lm-evaluation-harness** (EleutherAI) for every static benchmark above: 200+ YAML-defined tasks, the Korean
  tasks included, local Hugging Face checkpoints via `--model hf`, log-likelihood and generative tasks, per-task
  standard errors. It is the default choice of the field for this kind of static scoring.
- **mteb** for embedding models.
- **An own runner** for decision models: the recipes' `DecisionMethod.evaluate` already computes calibrated
  accuracy, NLL and ECE with and without reasoning; it needs a JevBench reader and a latency measurement.
- Not needed now: Inspect AI (agentic and tool-use evaluation), HELM (multi-metric profiling of many models),
  LLM-as-judge suites (a judge adds cost and bias that a checkpoint comparison does not need).

### Pitfalls the design must handle

- **Contamination.** A fine-tune on paraphrased benchmark items reaches inflated scores and evades n-gram checks
  (fine-tuning-phase contamination is the hardest to detect). The channel records, per evaluation, the overlap
  between the training rows of the experiment and the benchmark items (exact and near-duplicate), and the report
  shows it. The Data channel's leak filter is the place to prevent it; this channel only measures it.
- **Settings decide the number.** Few-shot count, prompt format, decoding (greedy vs sampled, temperature, max
  tokens), thinking on or off, harness version: each changes scores by points. Every result carries all of them,
  and two results are only compared when they match.
- **Small samples, noisy scores.** 231 JevBench items or a `limit` of 200 give confidence intervals of several
  points. The report shows the harness's standard error (or a bootstrap interval) next to every score, and marks a
  difference inside the interval as "not decided", not as "no difference".
- **Thinking and non-thinking are two models.** The decision recipes and Qwen3.5 answer differently with reasoning
  on. They are reported as separate rows, never averaged.
- **Base model first.** An evaluation of a checkpoint without the base model under the same settings is not
  interpretable; the compare stage insists on a baseline row.

## Design

A channel is a package that registers recipes; it reuses `Stage`, `Pipeline`, `Progress`, `Experiment` and the
API's jobs as they are, and joins by adding `"evaluation"` to `CHANNELS` in `core/registry.py`.

### Recipes

| Recipe | Input | Stages | Output |
|---|---|---|---|
| `evaluation_benchmark` | one checkpoint (or a Hub model) and a list of benchmarks | `score` (lm-evaluation-harness, one run per benchmark, cached by checkpoint content + task + settings + harness version) → `report` | `results/<benchmark>.json` with scores, standard errors, settings, samples; `report.json` |
| `evaluation_custom` | one checkpoint, any row source, the training recipe whose `evaluate` applies (`llm_sft`, `llm_dpo`, `llm_decision_sft`, ...) | `rows` → `score` → `report` | the recipe's own metrics on rows it never trained on |
| `evaluation_decision` | a decision checkpoint and JevBench or Jev-format rows | `rows` → `score` (with and without reasoning; accuracy, NLL, ECE, latency per item) → `report` | the Jeeves-style table |
| `evaluation_embedding` | an embedding checkpoint and MTEB task names | `score` (mteb) → `report` | MTEB results |
| `evaluation_compare` | several finished evaluation experiments (or checkpoints to evaluate with one benchmark list), the base model among them | `collect` → `report` | one table across checkpoints, a polygon chart per axis, a Markdown summary |

Config shape, following the modeling recipes (sections reject unknown keys; identities hash content):

```yaml
pipeline:
  recipe: evaluation_benchmark
  name: sft-qwen3.5-0.8b-eval
model:
  checkpoint: output/sft-qwen3.5-0.8b-<hash>/validate/checkpoint   # or name + revision for a Hub model
  thinking: false
benchmarks:
  - {name: mmlu_redux, num_fewshot: 5}
  - {name: gsm8k, num_fewshot: 8, limit: 500}
  - {name: ifeval}
  - {name: kmmlu, num_fewshot: 5, limit: 1000}
decoding: {temperature: 0, max_new_tokens: 1024}
contamination: {training_rows: output/sft-qwen3.5-0.8b-<hash>/data/train.jsonl}
axes: {knowledge: [mmlu_redux], math: [gsm8k], instructions: [ifeval], korean: [kmmlu]}
```

```yaml
pipeline:
  recipe: evaluation_compare
  name: sft-qwen3.5-0.8b-vs-base
runs:
  - {label: base, checkpoint: Qwen/Qwen3.5-0.8B, revision: <commit>}
  - {label: step-200, checkpoint: output/sft-qwen3.5-0.8b-<hash>/train/checkpoint}
  - {label: final, checkpoint: output/sft-qwen3.5-0.8b-<hash>/validate/checkpoint}
benchmarks: [...]        # the same list for every run; each run's scores come from the benchmark stage cache
```

### Where results live and how they are found

- Every `score` stage directory is `output/_stages/score-<fingerprint>/`, shared by every evaluation that asks for
  the same checkpoint, benchmark and settings: evaluating three checkpoints and then comparing them builds nothing
  twice, and re-running a comparison after one more checkpoint scores only that one.
- The checkpoint counts by content (`directory_signature`), as `model.init` does in training: a rebuilt experiment
  is a new evaluation.
- `report.json` per evaluation experiment: `{"checkpoint", "settings", "scores": {benchmark: {metric, stderr, n}},
  "contamination": {...}}`; the compare report adds `{"runs": [...], "axes": {axis: {label: value}}, "chart": "axes.svg"}`.
- Later: `GET /api/evaluations` listing reports across `output/`, and the tracker sending the table to wandb.

### Chart

One polygon (radar) per comparison: axes from `axes`, each axis the mean of its benchmarks' primary metric scaled
to 0..1, one polygon per run. Drawn as SVG by the channel itself (no plotting dependency); the Markdown summary
embeds it and the table.

## Plan

Each milestone ends green on CI with tests on the tiny random model, as the modeling channel does.

| Milestone | Builds | Done when |
|---|---|---|
| 1. Skeleton and `evaluation_custom` (built) | the `evaluation` package, `EvaluationConfig`, `rows` and `score` stages that call a modeling method's `evaluate` on a checkpoint, `report.json`, `CHANNELS` entry | `llmclick run configs/evaluation/custom.yaml` scores an SFT checkpoint on a JSONL file; the API lists the recipe; cached rerun builds nothing |
| 2. `evaluation_benchmark` (built; the GPU run is still to do) | lm-evaluation-harness behind an optional extra (`uv sync --extra eval`), a `benchmarks` section with presets (`small_general`, `korean`), harness version in the fingerprint, `limit` for smoke runs, settings recorded in results | the preset runs on Qwen3.5-0.8B on one GPU with `limit`; results carry stderr and settings; a missing extra gives one clear error |
| 3. `evaluation_compare` | `collect` over several checkpoints, the base model required, the table, the polygon SVG, Markdown summary, "not decided" marking from intervals | one command compares base, mid-training and final checkpoints of one experiment |
| 4. Decision and embedding | `evaluation_decision` (JevBench reader, think and no-think rows, ECE, latency), `evaluation_embedding` (mteb) | the decision pointer pilot reproduces the metrics `validate` reports, plus the JevBench tiers; the embedding pilot runs one MTEB retrieval task |
| 5. Contamination and surfacing | exact and near-duplicate overlap between an experiment's training rows and the benchmark items, in the report; `GET /api/evaluations`; tracker table | a deliberately contaminated pilot shows the overlap; the API returns the reports |

Order of 4 and 5 can swap depending on which models are trained first.

### Decisions to take before milestone 2

- **Panel.** Which of the general, Korean, code and long-context axes are needed for the models being trained.
  The default preset is general plus Korean, no code, no long context.
- **Dependency weight.** lm-evaluation-harness brings many dependencies; it goes behind an extra and is not in the
  Docker image unless the image is built with it.
- **Compute budget.** Full MMLU-Pro or KMMLU on a 2B model takes hours on one GPU; `limit` presets (smoke: 100,
  standard: 1000, full) are part of the config, and the report says which was used.
- **Thinking mode.** Whether the Qwen3.5 fine-tunes are evaluated with thinking on, off, or both (both doubles cost).

## Known starting points

- Each training method already has an `evaluate` that returns its metrics on a set of rows
  (`modeling/tuning/method.py`); milestone 1 calls it on rows that were never part of training.
- `DecisionMethod.collect` and `measured` (`modeling/llm/methods/decision/base.py`) compute accuracy, NLL and ECE
  with and without reasoning; milestone 4 adds a JevBench reader and timing around them.
- The removed jeff package carried benchmark builders (a five-benchmark panel and JevBench) for decision models;
  they are in the git history before the commit "vendored jeff 와 jev 레시피 제거".
- Row sources (`local_jsonl`, `huggingface`) live in `modeling/tuning/sources.py`; the `rows` stage reuses them.

## Sources

- Qwen3.5 small models and their benchmark table: [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models), [Qwen/Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- lm-evaluation-harness task list (Korean tasks kmmlu, haerae, kobest, click, hrm8k; mmlu_pro, gsm8k, ifeval, gpqa, humaneval, ruler, longbench): [tasks README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md)
- Harness comparison: [LLM evaluation tools comparison (2026)](https://inference.net/content/llm-evaluation-tools-comparison/), [lm-evaluation-harness review](https://www.aievals.co/tools/lm-evaluation-harness), [LightEval tutorial](https://cohorte.co/blog/lighteval-deep-dive-hugging-faces-all-in-one-framework-for-llm-evaluation)
- Korean benchmarks: [KMMLU](https://arxiv.org/abs/2402.11548), [KMMLU-Redux and KMMLU-Pro](https://arxiv.org/abs/2507.08924), [HAE-RAE Bench](https://arxiv.org/pdf/2309.02706), [Navigating Korean LLM research: evaluation tools](https://huggingface.co/blog/amphora/navigating-ko-llm-research-2)
- Contamination and settings: [systematic review of contamination detection (ACL 2026)](https://aclanthology.org/2026.gem-main.50/), [contamination survey, static to dynamic](https://arxiv.org/html/2502.17521v2), [DICE: contamination in the fine-tuning phase](https://arxiv.org/pdf/2406.04197), [Questionable practices in machine learning](https://arxiv.org/pdf/2407.12220)
- Leaderboards in 2026 (Open LLM Leaderboard archived; HELM, LMArena, Artificial Analysis): [LLM leaderboard explained](https://futureagi.com/blog/llm-leaderboard-explained/), [HELM](https://crfm.stanford.edu/helm/)
- Decision models: [PostHog Jeeves repository](https://github.com/PostHog/jeeves) (JevBench tiers, transfer panel, ECE, latency), [Jeeves announcement coverage](https://aiweekly.co/alerts/posthog-ships-jeeves-9b-decision-model-beating-jev-on-jevbench)
- Embeddings: [MTEB v2](https://huggingface.co/blog/isaacchung/mteb-v2), [Qwen3 Embedding](https://arxiv.org/pdf/2506.05176)

# Data channel

Status: designed, not built. This directory holds no code yet; this file is the design and the plan to build it,
written after a short survey of open datasets, synthetic-data methods and tooling (October 2026). The survey's
sources are at the end.

## Purpose

Produce the rows the Modeling channel trains on, two ways:

- **Open data**: load public datasets (Hugging Face Hub at a pinned revision, or local files), convert them into the
  row format of a recipe, filter them, remove what would leak into evaluation, and write JSONL.
- **Synthetic data**: generate rows with a teacher model (an API model or a local checkpoint) from seeds (personas,
  topics, documents, existing rows), verify them (schema, verifier, judge), filter, remove leaks, and write JSONL.

Either way the product is a JSONL file in a recipe's row format plus a manifest (counts per step, sources,
revisions, licences, teacher and cost), which Modeling reads through `data.sources` and never needs to know how it
was built. Leak removal against evaluation sets and benchmarks lives here as the last stage of both recipes; the
Evaluation channel only measures contamination, this channel is where it is prevented.

## Survey

**Assumptions.** Targets are the recipes of this repository (SFT, instruction, DPO, GRPO, decision, contrastive) on
0.6B to 2B models, English and Korean, one GPU or a Mac for local teachers and an API for stronger ones. Survey
depth: light.

### Open datasets by recipe

| Recipe | Dataset | Size | Why | Licence (check before use) |
|---|---|---|---|---|
| `llm_sft` | Tulu 3 SFT mixture (allenai) | 939k | seven domains, the reference open post-training mix | ODC-BY, per-source terms |
| `llm_sft` | SmolTalk / smol-smoltalk / smoltalk2 (HuggingFaceTB) | 1M / smaller | built for small models; smol-smoltalk for under 1B; Magpie-generated core | Apache-2.0 |
| `llm_sft` (reasoning) | OpenThoughts3-1.2M | 850k math, 250k code, 100k science | the published recipe for reasoning SFT (question sourcing, teacher traces, filtering) | Apache-2.0 |
| `llm_instruction`, Korean | KoAlpaca v1.1, KIT-19, open-korean-instructions, Magpie-Pro-MT-300K-ko, GLAN-QnA-KR | 21k to 303k | Korean instruction data is mostly translated or aggregated; GLAN-QnA-KR and Magpie-ko are native synthetic | varies |
| `llm_dpo` | UltraFeedback (binarized) | 64k | the standard preference set; SmolLM2 used it after SFT | MIT |
| `llm_dpo` | HelpSteer3-Preference (NVIDIA) | 40k | human-annotated, multi-domain and multilingual, commercial use allowed | CC-BY-4.0 |
| `llm_dpo`, Korean | Ko-UltraFeedback | 62k | translated and refined UltraFeedback | check |
| `llm_grpo` | GSM8K, MATH | 8k, 12k | verifiable answers; still discriminative for small models | MIT |
| `llm_grpo` | OpenR1-Math-220k, Open-Reasoner-Zero (129k) | 220k, 129k | larger verifiable math sets used by R1 replications | Apache-2.0 |
| `llm_grpo`, Korean | HRM8K | 8k | Korean math with answers | check |
| `embedding_contrastive` | MS MARCO / mMARCO, MIRACL (ko: 12.8k train pairs), BGE training collection, RLHN (relabelled hard negatives) | 600k+ | query-positive pairs; hard negatives mined or relabelled | varies |
| `embedding_contrastive`, Korean | ko-triplet-v1.0, korean-embedding-performance-v1 (1M) | 1M | Korean triples, mostly synthetic | check |
| `llm_decision_*` | the Jeeves prep sources: banking77, boolq, ag_news, mnli, sst5, yelp, trec, dbpedia14, amazon reviews, imdb, wanli (trainable); mmlu, emotion, tweet_eval offensive, qnli, paws, sciq (eval only), at the revisions `prep/public.py` pins | 4.7k accepted per source | classification and NLI datasets rendered as Jev questions, the reference decision-training data | per dataset |

### Synthetic-data methods

- **Self-Instruct / Evol-Instruct**: a seed set of instructions, evolved by a teacher into harder or more varied ones;
  the oldest recipe and still the baseline.
- **Magpie**: an instruct model given only its own chat prefix writes both the user turn and the answer; 300k
  filtered Magpie rows beat ShareGPT, Evol-Instruct and OpenHermes for Llama-3-8B, and SmolTalk's core is Magpie.
- **Persona Hub**: a billion web-derived personas as the diversity source; a persona in the prompt makes the same
  task yield different rows, the main defence against collapse of synthetic sets onto a few styles.
- **Distilled reasoning traces** (OpenThoughts): source questions, have a strong teacher write traces, keep by
  difficulty and verifier agreement; the mix (math, code, science) was tuned by ablation.
- **Rejection sampling with verifiers**: sample several answers, keep those a verifier accepts (math answers, unit
  tests); doubles as preference pairs (accepted vs rejected) for DPO.
- **Judge-ranked preferences** (UltraFeedback): several models answer, a judge scores by aspect, the best and a
  worse answer form the pair.
- **Multi-agent simulation** (MATRIX-Gen): personas interact to create realistic multi-turn conversations.
- **Document-grounded QA** (synthetic-data-kit, Easy Dataset): chunk documents, generate questions, answers and
  traces per chunk; the route for domain data.
- **Decision questions over states** (jeff): a teacher writes typed questions (choice, noul, score) with answers
  about a given state text, producing Jev records from any corpus of states.

### Tools

- **distilabel** (Argilla): pipeline framework with steps for EvolInstruct, UltraFeedback, Magpie and judges; the
  most complete open implementation of the methods above.
- **Bespoke Curator**: batch-oriented generation with structured outputs, used for reasoning and function-calling data.
- **synthetic-data-kit** (Meta): documents in, QA and CoT pairs out, with a CLI.
- **Anthropic API** as a teacher: `claude-opus-5-5` by default, the Message Batches API at half price for bulk
  generation, structured outputs (`output_config.format`) to get rows that already match a recipe's schema, prompt
  caching for the shared instructions and persona lists.
- **Local teacher**: any checkpoint the Modeling channel can load, through `LLMBackbone.generate`; free and
  reproducible (seeded), weaker.

The channel does not adopt distilabel or Curator as a dependency: their pipelines overlap with the `Stage` and
`Pipeline` runner already here (fingerprints, caching, resume, progress, jobs), and a generation step is a few
hundred lines on top of the Anthropic SDK and the existing backbones. Their method implementations are the
reference for the prompts and the filters.

### Quality, duplication, leakage

- **Near-duplicates**: MinHash over word 5-grams with LSH (the FineWeb setting: 112 hashes in 14 buckets, 75%
  similarity) removes paraphrased repeats that exact hashing misses; embedding-based dedup catches semantic repeats.
- **Decontamination**: remove training rows whose n-gram overlap with an evaluation item exceeds a threshold (50%
  MinHash similarity, or answer overlap at 60% and passage overlap at 40% in token-level detectors). The Evaluation
  channel measures exactly this kind of overlap after the fact; this channel removes it before training.
- **Judges**: a hostile judge from a different model family than the generator, scoring faithfulness, clarity,
  plausibility and novelty against the task definition, rejects weak rows; learned quality filters beat heuristics
  for educational value and toxicity.
- **Diversity and collapse**: filtering and dedup are what keeps the effective distribution of synthetic data broad;
  seed diversity (personas, documents) matters more than volume. Human spot-checks on a sample stay in the loop.
- **Known pitfalls**: contamination introduced through instruction tuning is the hardest to detect, and paraphrased
  benchmark items evade n-gram detectors, so the leak stage must also run on synthetic rows, with the teacher's
  tendency to reproduce benchmark items in mind. Translated Korean corpora carry translation artefacts; native
  synthetic generation with Korean personas is the better route for Korean SFT. Licences differ per source and
  must be recorded per row set.

## Reference practices for synthetic quality

What the published recipes actually did, with the numbers they report; the checklist at the end is what
`data_synthetic` builds in. Sources are at the end of this file.

**Seeds decide coverage; the prompt decides little.**
- Taxonomy-driven (GLAN): fields, sub-fields, disciplines, then subjects and a syllabus per subject, generated
  with LLM help and reviewed; instructions come from the syllabus, no seed data. GLAN-QnA-KR reproduces it in
  Korean over 1,084 disciplines with a 100 to 900 difficulty scale and a two-layer contamination audit against
  Korean benchmarks.
- Persona-driven (Tulu 3, Persona Hub): about 250k personas from Persona Hub combined with per-skill prompts
  (precise instruction following, math, code) so one task yields many perspectives; Persona Hub holds a billion
  web-derived personas.
- Document-driven (Phi-4): high-quality seed passages from many domains rewritten into exercises, discussions
  and reasoning tasks by multi-step prompting; instruction reversal (from an answer, write the instruction) adds
  instruction data from any text; about 50 synthetic dataset types, 400B tokens.
- From nothing (Magpie): an instruct model given only its chat prefix writes the user turn and then the answer;
  cheap and diverse, but everything rests on the filtering below.

**Questions: few good sources, many answers.** OpenThoughts (27 code, 21 math, 14 science sources ablated):
mixing at most two high-quality question sources beat mixing sixteen by 5 points on average; sampling 16
answers per question beat adding questions (answer diversity inside a problem is worth more than question
diversity across problems); filtering answers by majority consensus or length did not beat keeping every sample;
the best benchmark model was not the best teacher (QwQ-32B taught better than DeepSeek-R1). The teacher and the
sources are chosen by training a student and measuring, not by the teacher's own scores.

**Filtering is where the quality is made.**
- Magpie's filters on 4M raw rows down to 300k: input quality tagged by a judge (keep at least average or good),
  input difficulty (at least medium), a reward model score and the reward difference against the base model's
  answer (at least 0), removal of repetition and incomplete instructions, minimum neighbour distance in embedding
  space (all-mpnet-base-v2 with FAISS) against near-duplicates, then the longest responses up to the budget.
- SmolTalk's core (Smol-Magpie-Ultra, 400k) was generated with Llama-3.1-405B-Instruct and filtered with smaller
  Llama judges plus the ArmoRM reward model; for small models the mix adds task-specific sets (constraints 36k,
  rewrite 50k, summarize 100k) and a reduced `smol-smoltalk` for models under 1B.
- Nemotron-4 340B gated 98% synthetic post-training data with a five-attribute reward model (helpfulness,
  correctness, coherence, complexity, verbosity) for both filtering and preference ranking, iterated weak to strong.
- Phi-4 added self-revision: the model critiques and improves its own answers against rubrics on reasoning and
  factual accuracy, then rejection sampling for post-training.
- Tulu 3 made instruction following verifiable: 25 constraint types from IFEval, about 30k examples checked by
  heuristics, no judge needed.

**Selection scorers are cheap filters, not oracles.** Deita (complexity and quality scorers plus a diversity
constraint), IFD (instruction-following difficulty from a small model's perplexity with and without the
instruction; 5% of Alpaca selected by IFD beat the full set), AlpaGasus (judge scores on helpfulness and
accuracy), InsTag (tag diversity). At scale, random selection is hard to beat with these alone, so a scorer is
used to drop the worst rows and the rest is decided by training a student.

**Decontamination and duplicates.** Tulu 3 matched evaluation and training by shared 8-grams per token and dropped
the overlaps; GLAN-QnA-KR audits against the Korean benchmarks; MinHash with LSH (5-grams, 112 hashes, 75%)
removes paraphrased repeats; embedding clustering shows whether a large set is really a few clusters. Synthetic
rows need this as much as scraped ones: teachers reproduce benchmark items from memory, and paraphrases evade
n-gram checks, so the leak stage runs on every synthetic set and the Evaluation channel reports what remains.

**Collapse is avoided by accumulation and diversity, not by volume.** Training on synthetic data that replaces
real data collapses (as little as one synthetic row in a thousand under replacement, per ICLR 2025 work);
accumulating synthetic rows on top of real ones bounds the error (Stanford SALT, 2024). The tail still erodes
inside the synthetic share, so diversity is measured (Vendi score, number of embedding clusters, the minimum
neighbour distance distribution) and generation stops when it falls, rather than when a row count is reached.

**The only ground truth is a trained student.** SmolLM2, Tulu 3 and OpenThoughts all chose data by training a
model on it and evaluating; a 5 to 10% judge audit of a new set (the LLM Data Auditor framing: diversity,
correctness, instruction adherence, safety) decides whether to filter row by row before that.

**Korean.** Translated corpora carry artefacts (KIT-19's finding, and the reason GLAN-QnA-KR and Magpie-ko generate
natively); the recipe is native generation from Korean personas or a Korean-labelled taxonomy, a Korean rubric
for the judge, and decontamination against KMMLU, HAE-RAE, KoBEST and CLIcK.

### What `data_synthetic` builds in

- Seeds as first-class input: taxonomy, personas, documents, existing rows; a run records which seeds produced
  which rows, so coverage can be read off the manifest.
- Questions from at most a few sources per run, and `answers_per_question` as a setting (default above 1 for
  reasoning and preference tasks); the teacher is a config choice to be compared by a student's evaluation.
- The Magpie filter chain as named filters with thresholds in the config: schema, completeness and repetition,
  judge tags (quality, difficulty), reward or judge score with a margin against a base answer, minimum neighbour
  distance, length policy. Rejections are kept with reasons.
- Verifiers before judges: heuristics for constraints, `REWARDS` for answers and code, a judge from another
  model family last. Self-revision as an optional second pass with a rubric.
- Leak removal against the evaluation sets and benchmarks the model will be measured on, in every run.
- Diversity numbers in the manifest (clusters, neighbour distances, Vendi score) and a stop rule on them.
- The student loop: a sample config that trains the pilot recipe on the new rows and runs `evaluation_compare`
  against the base model, so a data run ends with a measured verdict rather than a row count.

## Design

A channel is a package that registers recipes; it reuses `Stage`, `Pipeline`, `Progress`, `Experiment` and the
API's jobs as they are, and joins by adding `"data"` to `CHANNELS` in `core/registry.py`.

### Recipes

| Recipe | Input | Stages | Output |
|---|---|---|---|
| `data_open` | public or local datasets, the target recipe, a converter, filters, leak targets | `load` -> `convert` -> `filter` -> `leak` -> `write` | `rows.jsonl` in the recipe's format, `manifest.json` |
| `data_synthetic` | a teacher, seeds, a prompt and task kind, verification, filters, leak targets | `seeds` -> `generate` -> `verify` -> `filter` -> `leak` -> `write` | the same |

```yaml
pipeline: {recipe: data_open, name: tulu-sft-ko}
sources:
  - {name: huggingface, dataset: allenai/tulu-3-sft-mixture, revision: <commit>, split: train, limit: 50000}
convert: {to: llm_sft, converter: messages}                 # or fields: {messages: conversation}
filters:
  length: {min_chars: 20, max_chars: 8000}
  languages: [en, ko]
  dedup: {method: minhash, ngram: 5, threshold: 0.75}       # exact is always on
leak:
  against: [{name: local_jsonl, path: output/sft-qwen3.5-0.8b-<hash>/data/validation.jsonl}, {benchmark: gsm8k}]
  ngram: 8
  threshold: 0.5
```

```yaml
pipeline: {recipe: data_synthetic, name: persona-chat-ko}
teacher: {name: anthropic, model: claude-opus-5-5, batch: true, max_tokens: 2048, temperature: 1.0}
task: {kind: conversation, to: llm_sft, turns: [2, 6]}
seeds:
  - {name: personas, source: {name: huggingface, dataset: proj-persona/PersonaHub, revision: <commit>, split: train, limit: 2000}}
  - {name: topics, values: [customer support, travel, cooking]}
prompt: {template: prompts/persona_chat_ko.jinja, language: ko}
count: 5000
verify:
  schema: true                                               # structured outputs: the row must parse as the recipe's row
  judge: {teacher: {name: anthropic, model: claude-sonnet-5-5}, rubric: prompts/judge_helpfulness.jinja, min_score: 7}
filters: {dedup: {method: minhash, threshold: 0.75}}
leak: {against: [{benchmark: kmmlu}, {benchmark: haerae}]}
budget: {max_usd: 50}
```

- **Converters** are a registry (`CONVERTERS`): `messages` (a column of chat turns), `instruction` (instruction,
  input, output columns), `pairs` (prompt, chosen, rejected; UltraFeedback and HelpSteer3 shapes), `answer`
  (prompt and a verifiable answer; GSM8K's `#### 18`), `triples` (query, positive, negatives; MIRACL and MS MARCO
  shapes), `classification` (a text and a label set rendered as a Jev `choice` question with a per-source template,
  as the Jeeves prep does; NLI as `choice`, boolean as `noul`, ratings as `score`). A converter has a `version`
  that joins the fingerprint. `fields` maps columns when no converter fits.
- **Teachers** are a registry (`TEACHERS`) with four entries from the start: `anthropic` (the official SDK;
  Message Batches when `batch: true`; structured outputs for schema verification; prompt caching on the template;
  the key from `ANTHROPIC_API_KEY` through settings), `openai` (the OpenAI chat-completions format: OpenAI itself,
  vLLM, LM Studio, any gateway with that shape; `base_url` and `OPENAI_API_KEY`), `ollama` (a local Ollama server
  through its native API, with model pull and `keep_alive`; its OpenAI-compatible endpoint works through `openai`
  too), and `local` (a checkpoint through the Modeling channel's backbones, seeded, no cost). Every teacher offers
  the same `complete(messages, schema?)` and reports tokens and an estimated cost per call into the manifest;
  `budget.max_usd` stops a run before it overspends. JSON-schema output goes through each provider's native
  structured-output feature where it exists and through a parse-and-retry otherwise.
- **Verification** chain: `schema` (the row parses as the target recipe's row, through the recipe's own `check`),
  `verifier` (a reward function from `REWARDS`, for answers and code), `judge` (a teacher scoring with a rubric; a
  different model family than the generator by default). Rejected rows are kept in `rejected.jsonl` with the reason.
- **Leak removal** reuses `evaluation/contamination.py` (exact and n-gram overlap) and the `jevbench` and benchmark
  readers for items, so the same arithmetic that measures contamination removes it; `against` takes row sources,
  benchmark names (through lm-evaluation-harness's documents) and JevBench tiers.
- **Fingerprints**: dataset revisions and file contents, converter versions, filter settings, the prompt template's
  content, the teacher's model and generation settings and the seed. `generate` is resumable: rows are written
  per seed batch as they arrive, and a rerun continues from the last batch; the cache directory makes a second
  config with the same seeds and prompt reuse the generation.
- **Output**: `output/<experiment>/write/rows.jsonl` and `manifest.json` (per stage the counts in and out,
  sources with revisions and licences, teacher, tokens, cost, rejected reasons). Modeling's `data.sources` points
  at the rows; later a `dataset` source can name a data experiment directly.
- **Row sources move here**: `local_jsonl` and `huggingface` in `modeling/tuning/sources.py` become
  `data/sources.py`; Modeling and Evaluation import them from here.

### Reports and surfacing

`manifest.json` is the report. `GET /api/datasets` lists manifests under an output directory (counts, sources,
cost); with `tracker.enabled` the manifest's numbers go to wandb as the evaluation reports do.

## Plan

Each milestone ends green on CI with tests on small local files and a fake teacher, as the other channels do.

| Milestone | Builds | Done when |
|---|---|---|
| 1. Skeleton and `data_open` | the `data` package, row sources moved here, `load`, `convert` (`messages`, `instruction`, `pairs`, `answer`, `triples`, `fields`), `filter` (length, exact dedup, language), `leak` (reusing the contamination code), `write` with the manifest; sample configs for Tulu, UltraFeedback, GSM8K, MIRACL | `llmclick run configs/data/open_sft.yaml` writes rows that `configs/llm/sft.yaml` trains on by path; a rerun builds nothing; the API lists the recipe |
| 2. Decision rows and near-duplicates | the `classification` converter with per-source templates for the Jeeves prep sources (trainable and eval-only panels), MinHash near-duplicate removal, the Jeeves transfer panel as decision evaluation rows for `evaluation_decision` | a decision SFT pilot trains on converted banking77 rows; the transfer panel scores a decision checkpoint |
| 3. `data_synthetic` with a local teacher | `seeds`, `generate` (resumable, per-batch cache, seeded), `verify` (schema and verifier), the `local` teacher, instruction and conversation tasks (Magpie-style and persona-seeded prompts), templates | a tiny local model generates rows that `llm_sft` trains on; the fake teacher tests cover resume and cost accounting |
| 4. Anthropic teacher, judge, preferences and traces | the `anthropic` teacher (Batches, structured outputs, caching, budget), the `judge` verifier, preference pairs by rejection sampling and by judge, reasoning traces with verifier agreement, decision questions over states | a 1k-row Korean persona conversation set and a 1k-pair preference set are produced within the budget and pass the judge; costs are in the manifest |
| 5. Korean presets and surfacing | presets for Korean SFT, DPO and retrieval rows, translation-artefact filters, `GET /api/datasets`, the tracker table, a `dataset` row source for Modeling | the presets run end to end on a GPU; the API lists the manifests |

### Decisions to take before milestone 3

- **Default teacher.** The Anthropic API (`claude-opus-5-5`; `claude-sonnet-5-5` for judges and bulk) or a local
  checkpoint. The design supports both; the default decides what the sample configs show and what a run costs.
- **Budget per run.** `budget.max_usd` default and whether Batches (half price, hours of latency) is the default.
- **First target.** Korean conversation rows for `llm_sft`, decision rows for `llm_decision_sft`, or preference
  pairs for `llm_dpo`; the milestones above assume SFT first, decision second.
- **Licence policy.** Which licences are acceptable for training rows (commercial use or not); the manifest
  records them either way.

## Known starting points

- Row sources (`local_jsonl`, `huggingface`) in `modeling/tuning/sources.py`, and the `jevbench` source in
  `evaluation/decision.py`.
- Contamination measurement in `evaluation/contamination.py` (exact and n-gram overlap, numpy-backed), to be
  reused as the leak filter.
- Reward functions in `modeling/llm/methods/grpo.py` (`exact_match`, `contains`, `last_number`) as verifiers.
- The vendored jeff package that was removed carried a data pipeline (27 public datasets converted into decision
  rows, teacher-written synthetic questions, a near-duplicate leak filter). It is in the git history before the
  commit "vendored jeff 와 jev 레시피 제거" and is the reference for the decision converters and the question
  generator; the Jeeves repository's `prep/` (public sources at pinned revisions, admission rules) is the other.

## Sources

- Synthetic data methods and tools: [Distilabel](https://github.com/argilla-io/distilabel), [Bespoke Curator and synthesis landscape (2026 guide)](https://www.premai.io/blog/how-to-generate-synthetic-training-data-for-llm-fine-tuning-2026-guide/), [Meta synthetic-data-kit](https://github.com/meta-llama/synthetic-data-kit), [Scaling synthetic data creation with 1B personas](https://arxiviq.substack.com/p/scaling-synthetic-data-creation-with), [MATRIX-Gen: multi-agent simulation](https://arxiv.org/pdf/2410.14251), [Easy Dataset](https://arxiv.org/pdf/2507.04009)
- SFT and reasoning sets: [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), [SmolLM2 paper](https://arxiv.org/pdf/2502.02737), [OpenThoughts](https://arxiv.org/pdf/2506.04178)
- Preference sets: [UltraFeedback and a systematic study of DPO datasets](https://arxiv.org/pdf/2511.10985), [HelpSteer3-Preference](https://arxiv.org/html/2505.11475v1), [curated list of post-training datasets](https://github.com/mlabonne/llm-datasets)
- Verifiable reasoning sets: [100 days after DeepSeek-R1 (replication survey)](https://arxiv.org/pdf/2505.00551), [training small R1-like models](https://www.stephendiehl.com/posts/small_reasoning_models)
- Korean: [KIT-19](https://arxiv.org/html/2403.16444), [GLAN-QnA-KR](https://arxiv.org/html/2607.20443), [open-korean-instructions](https://huggingface.co/datasets/heegyu/open-korean-instructions), [KoAlpaca](https://github.com/Beomi/KoAlpaca)
- Embedding training data: [MIRACL and multilingual retrieval data engineering](https://arxiv.org/pdf/2302.07010), [korean-embedding-performance-v1](https://huggingface.co/datasets/LLM-OS-Models/korean-embedding-performance-v1-performance-1m), [embedding models 2026 guide](https://tensoria.fr/en/blog/embedding-models-2026-guide)
- Quality, dedup, decontamination: [FineWeb-Edu](https://www.emergentmind.com/topics/fineweb-edu-dataset), [LSHBloom](https://arxiv.org/html/2411.04257v3), [NVIDIA text data processing](https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/), [contamination detection review (ACL 2026)](https://aclanthology.org/2026.gem-main.50/)
- Decision data: [PostHog Jeeves repository (`prep/`, `data/manifest.json`)](https://github.com/PostHog/jeeves)
- Reference practices: [Magpie (ICLR 2025)](https://arxiv.org/html/2406.08464) and its [filtered 300k sets](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered), [OpenThoughts findings](https://www.openthoughts.ai/blog/ot3), [Tulu 3 report](https://arxiv.org/pdf/2411.15124), [Phi-4 technical report](https://arxiv.org/html/2412.08905v1), [Nemotron-4 340B report](https://arxiv.org/pdf/2406.11704), [GLAN](https://arxiv.org/pdf/2402.13064), [Deita and the data-selection survey](https://arxiv.org/pdf/2402.05123), [Superfiltering (IFD)](https://www.researchgate.net/publication/384220620_Superfiltering_Weak-to-Strong_Data_Filtering_for_Fast_Instruction-Tuning), [random selection at scale](https://arxiv.org/html/2410.09335v1), [LLM Data Auditor survey (2026)](https://arxiv.org/html/2601.17717v1), [Best practices and lessons learned on synthetic data](https://arxiv.org/pdf/2404.07503), [model collapse under accumulation](https://alphaxiv.org/overview/2410.16713v4), [synthetic data pipelines that don't collapse](https://tianpan.co/blog/2026-04-12-synthetic-data-pipelines-that-dont-collapse)

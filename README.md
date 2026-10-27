# LLMClick

Documentation: https://ash-hun.github.io/LLMClick/ (generated from this README, the channel READMEs and the code; see `website/README.md`).

Config-driven custom model building. One YAML describes one custom model; `pipeline.recipe` picks the recipe that
builds it and `pipeline.stages` picks the steps to run. Every recipe runs on the same pipeline runner, which skips
what is already built, shows progress, and never hands on a trained model that has not passed validation.

| Family | Recipe (`pipeline.recipe`) | Trains | Model catalog (`model.architecture`) |
|---|---|---|---|
| LLM | `llm_sft` | conversations, loss on every assistant turn | `transformer` (Qwen3-0.6B, 1.7B), `hybrid` (Qwen3.5-0.8B, 2B) |
| LLM | `llm_instruction` | instruction/input/output records | same |
| LLM | `llm_dpo` | chosen/rejected preference pairs | same |
| LLM | `llm_grpo` | prompts scored by a reward function | same |
| LLM | `llm_decision_sft` | decision questions: a head scores the options (Jev, jeff, Jeeves stage 1) | same, plus `model.head`: `readout` or `pointer` |
| LLM | `llm_decision_cispo` | the same questions with reasoning before the decision, by reinforcement (Jeeves stage 2) | same |
| Embedding | `embedding_contrastive` | query/positive/negative triples (InfoNCE) | `bi_encoder` (Qwen3-Embedding-0.6B) |
| TTS | `tts_piper` | recordings and their texts: a Piper (VITS) voice fine-tuned from a Piper checkpoint, exported to ONNX | `model.quality`: `medium`, `high` (Piper's sizes) |
| Wake word | `wakeword` | recordings of the agent's name ("Jarvis") and of everything else: an openWakeWord detector, exported to ONNX | openWakeWord's `dnn` head on its frozen speech features |
| Evaluation | `evaluation_custom` | nothing: scores a checkpoint of any experiment above on any rows, with that recipe's metrics | the experiment's |
| Evaluation | `evaluation_benchmark` | nothing: scores a checkpoint on public benchmarks (lm-evaluation-harness), one cached stage per benchmark | the experiment's (LLM) |
| Evaluation | `evaluation_compare` | nothing: several checkpoints (the base model among them) on the same benchmarks and rows; table, deltas with intervals, polygon chart | the experiments' |
| Evaluation | `evaluation_decision` | nothing: a decision checkpoint on Jev rows or the JevBench tiers, without and with reasoning; accuracy, NLL, ECE, per tier, latency | the experiment's (decision) |
| Evaluation | `evaluation_embedding` | nothing: an embedding checkpoint on MTEB tasks (mteb), one cached stage per task | the experiment's (embedding) |
| Evaluation | `evaluation_wakeword` | nothing: a wake word detector on recordings and hours of audio it never heard; miss rate per noise condition, false accepts per hour, the threshold sweep | the experiment's (wake word) |
| Data | `data_synthetic` | nothing: rows for the recipes above from seeds through a teacher model (seeds, prompts, evolve, respond, verify, select) | teachers: anthropic, openai, ollama, local |
| Data | `data_speech` | nothing: a Piper training set from seeds: texts (a teacher, or the seeds as they are) read by a Piper voice, checked by Whisper | voice: any Piper `.onnx`, one speaker or many |

The decision recipes follow the Jev request format. The readout head and one-pass training follow
[jeff](https://github.com/firelex/jeff); the pointer head, reasoning chains and CISPO follow
[Jeeves](https://github.com/PostHog/jeeves). Both are reimplemented on this framework's training loop, so their
released checkpoints do not load here.

## 01. Project structure

```
LLMClick/
├── main.py                       # `uvicorn main:app` from the repository root; the app itself is core/api/app.py
├── configs/                      # one YAML = one custom model; a directory per family (llm/, embedding/)
├── samples/                      # tiny rows per recipe so the shipped configs run offline
├── core/                         # the framework, shared by every channel
│   ├── stage.py                  #   Stage: one fingerprinted, rerunnable unit of work
│   ├── pipeline.py               #   Pipeline: runs a recipe's stages; build()/load() turn a config into one
│   ├── progress.py               #   Progress, BarProgress (terminal), StateProgress (API jobs)
│   ├── registry.py               #   Registry, RECIPES, CHANNELS
│   ├── config/                   #   schema.py (BaseConfig), experiment.py (directories, manifest)
│   ├── cli.py                    #   llmclick run|validate|recipes|serve
│   ├── settings.py               #   process settings from the environment and environment/.env (LLMCLICK_ENV)
│   ├── api/                      #   app.py; routers: config_channel, job_channel, system_channel; store.py, guard.py
│   └── utils/                    #   files (hash, lock), device
├── modeling/                     # Modeling channel: config -> trained, validated model
│   ├── config.py                 #   ModelingConfig: tracker, validation bounds
│   ├── stages.py                 #   TrainStage, ValidateStage (templates every recipe fills in)
│   ├── pipeline.py               #   ModelingPipeline: training always brings validation
│   ├── tracker.py                #   wandb replay
│   ├── tuning/                   #   in-process fine-tuning shared by the families below
│   │   ├── backbone.py           #     Backbone: base of every model catalog entry
│   │   ├── method.py             #     TrainingMethod: base of every training method
│   │   ├── loop.py               #     the one training loop (accumulation, schedule, resume)
│   │   ├── stages.py             #     data, train, validate
│   │   ├── sources.py            #     row sources: local_jsonl, huggingface
│   │   └── config.py pipeline.py #     TuningConfig, TuningPipeline
│   ├── llm/                      #   family: generative language models
│   │   ├── models/               #     Model Catalog: base.py (LLMBackbone), transformer.py, hybrid.py
│   │   │   └── heads/            #       decision heads on top of a backbone: base.py (DecisionHead), readout, pointer
│   │   ├── methods/              #     Training Method: base.py (LLMMethod), sft, instruction, dpo, grpo
│   │   │   └── decision/         #       base.py (DecisionMethod: rows, metrics, calibration), sft, cispo
│   │   └── config.py pipeline.py #     one config class and one registered recipe per method
│   ├── embedding/                #   family: text embedding models
│   │   ├── models/               #     Model Catalog: base.py (EmbeddingBackbone), bi_encoder.py
│   │   ├── methods/              #     Training Method: contrastive.py
│   │   └── config.py pipeline.py
│   ├── tts/                      #   family: text-to-speech (tts_piper)
│   │   ├── piper.py              #     the Piper program (GPL) as a separate process: speak, fit, export
│   │   ├── audio.py asr.py       #     WAV measurements; Whisper transcripts and word error rate
│   │   └── config.py stages.py pipeline.py
│   └── wakeword/                 #   family: wake word detection (wakeword)
│       ├── features.py           #     openWakeWord's frozen front end on onnxruntime: audio -> embeddings every 80 ms
│       ├── detector.py           #     the DNN head: trained with torch, exported and scored in ONNX
│       └── config.py stages.py pipeline.py
├── data/                         # Data channel: seeds -> prompts -> evolve -> respond -> verify -> select (data_synthetic)
│   ├── teachers.py               #   anthropic, openai, ollama, local behind one complete()
│   ├── config.py rows.py ...     #   one module per stage; rows.py: natural keys, atomic files, per-item caches
│   └── speech/                   #   data_speech: synthesize, verify, select after the shared seeds and prompts
├── evaluation/                   # Evaluation channel: config -> scores of a trained model, report.json
│   ├── source.py                 #   SourceExperiment: an experiment's config read back, its checkpoint located
│   ├── stages.py                 #   rows, score, report (evaluation_custom)
│   ├── benchmark.py              #   presets, one score stage per benchmark over lm-evaluation-harness (evaluation_benchmark)
│   ├── compare.py                #   runs, verdicts against the base, polygon SVG, Markdown summary (evaluation_compare)
│   ├── decision.py               #   jevbench source, decide without and with reasoning, ECE, tiers, latency (evaluation_decision)
│   ├── embedding.py              #   mteb encoder wrapper, one stage per MTEB task (evaluation_embedding)
│   ├── wakeword.py               #   clips, score traces, miss rate and false accepts per hour over thresholds (evaluation_wakeword)
│   └── config.py pipeline.py     #   (README: design and milestones)
├── tests/core/  tests/modeling/
├── docs/                         # api-spec.md, screen-spec.md
├── environment/                  # Dockerfile, docker-compose.yml, .env.sample
│   └── piper/                    #   the Piper program's own locked environment (GPL), built by setup.sh
└── output/
    ├── _stages/<stage>-<fingerprint>/   # what each stage built; shared by every experiment with the same inputs
    └── <name>-<hash>/                   # one experiment: config.yaml, manifest.json, pipeline.log, links to its stages
```

`evaluation/` scores what `modeling/` built: `evaluation_custom` measures any checkpoint of an experiment (`validate`,
`train`, `base`, or a directory) on any rows with the metrics of the recipe that trained it; `evaluation_benchmark`
scores it on public benchmarks through lm-evaluation-harness (`uv sync --extra eval`), one cached stage per
benchmark, with presets for the small-model panel and Korean; `evaluation_compare` puts several checkpoints, the
base model among them, through the same benchmarks and rows and reports the differences with their intervals, a
polygon chart and a Markdown summary; `evaluation_decision` measures a decision model on Jev rows or the JevBench
tiers without and with reasoning (accuracy, NLL, ECE, per tier against chance, latency); `evaluation_embedding`
runs an embedding model on MTEB tasks; `evaluation_wakeword` measures a wake word detector's miss rate under noise and
false accepts per hour of audio without the name. All write `report.json` with the overlap between the evaluation items and
the experiment's training rows (exact matches and n-gram near-duplicates) next to the scores, send the report to
wandb with `tracker.enabled`, and are listed by `GET /api/evaluations` (`evaluation/README.md` has the details). `data/` produces rows: `data_synthetic` makes them from seeds through a teacher model (Anthropic, OpenAI-format, Ollama or a local checkpoint), evolves and verifies them, removes duplicates and leaks, and writes a recipe's rows with a manifest; every stage is cached by its inputs and safe to rerun, so a crash or a spent budget resumes where it stopped (`data/README.md`). Loading open datasets (`data_open`) is designed there and not built. A channel is a
package that registers recipes; it joins by adding its name to `CHANNELS` in `core/registry.py` and reuses `Stage`,
`Pipeline`, `Progress` and `Experiment` as they are.

## 02. How a run works

```yaml
pipeline:
  recipe: llm_sft                   # which pipeline builds this model
  name: my-model
  stages: [data, train]             # optional; default all. Stages they read are added, train always brings validate
```

- **Fingerprints.** A stage's fingerprint is its own config inputs (plus the content of local files it reads) and
  the fingerprints of the stages it reads. Its directory is `output/_stages/<stage>-<fingerprint>/`, so two configs
  that differ only in training settings share the data stages, and a rerun of the same config builds nothing.
  Each training method, backbone and head class carries a `version`; bumping one after changing its code rebuilds
  the experiments that use that part and leaves every other recipe's cache alone.
- **Reruns.** A stage is recorded only when it finishes. A crashed run is resumed by running the same command
  again; training continues from `resume.pt` in the train stage directory and ends with the same weights as an
  uninterrupted run: batches, learning rates and the samples of a sampling method (`llm_grpo`, `llm_decision_cispo`)
  are all derived from the seed and the step. With `method.iterations` above 1 a snapshot is written only between
  groups of steps that share samples, so a resume never lands inside such a group.
- **Locks.** A stage directory is locked while it is built, so a CLI run and an API job asking for the same stage
  never build it twice; the second one says that it is waiting (progress note, log) and then reuses the result.
- **Identity.** The experiment directory is `<name>-<hash>`; `stages`, `output_dir` and `tracker` are not part of
  the hash, because they say how to run, not what to build.
- **Progress.** The CLI shows two bars (stages, and steps inside the running stage); API jobs report the same
  state in `progress`. Every long phase names itself there: loading the model, checking lengths, a method's
  preparation (reference margins, reasoning chains), the steps, fitting the temperature, saving.
- **Lengths.** Before the first training step every training and held-out row is measured against
  `training.max_length`, including the tokens a method adds later (generated completions, reasoning chains).
  A row that does not fit stops the run there (`training.overflow: error`, the default) or is left out
  (`overflow: skip`); nothing is cut silently and nothing fails hours into training. Embedding texts are the
  exception: they are truncated, as usual for retrieval.
- **Evaluation while training.** `training.eval_every: N` measures the model on the held-out rows every N steps
  with the validate stage's own metrics (`training.eval_rows` caps how many rows), and records them as
  `eval/<metric>` next to the loss in `training.jsonl` and in the tracker. The weights are as they are at that step:
  for the decision recipes the temperature is not fitted yet, so `nll` and `ece` there are uncalibrated.
- **Validation.** In the Modeling channel `validate` follows `train` and cannot be left out. It measures the
  trained model on data training never saw and checks `validation.min` / `validation.max`; a miss, or a metric
  that is not a finite number, stops the pipeline. A later stage reads the checkpoint from `validate`, never from
  `train`. Without bounds the metrics are only recorded, and a warning says so.
- **Tracking.** With `tracker.enabled` each training step is sent to wandb as it happens; a run interrupted and
  resumed continues the same wandb run. `tracker` is not part of an experiment's identity, so switching it on later
  replays the already cached run once instead of retraining. A run is sent to a given project only once.
- **Jobs.** The API keeps its jobs in a SQLite table (`JOBS_DB`), so they survive a restart; a job the server was
  running when it stopped is marked `interrupted` and resumes when submitted again. `DELETE /api/jobs/<id>` cancels
  a pending or running job: it stops at its next progress report, keeps what it built, and continues from there
  when submitted again. `JOB_WORKERS` jobs run at once, each worker on its own accelerator: set it to the number of
  GPUs on a multi-GPU server and keep 1 otherwise. On a server others can reach, set `API_TOKEN` (every `/api` call
  then needs `Authorization: Bearer <token>`) and `API_PATHS` (the directories the API may read configs and rows
  from and write output to; any other path is refused).
- **Typos.** Every config section rejects keys it does not know, so `validaton:` or `training.learning_rate` is an
  error instead of a silently ignored setting.

### Stages

| Stage | What it does |
|---|---|
| `data` | read every `data.sources[*]`, check each row against the recipe's method, hold out `data.validation` of them |
| `train` | load the backbone, run the training loop with the method's loss; resumes from `resume.pt` after a crash |
| `validate` | the method's metrics on the held-out rows against `validation.min`/`max` |
| `rows`, `score`, `report` | `evaluation_custom`: read and check the rows, measure the chosen checkpoint with the experiment's method, write `report.json` |
| `score:<task>`..., `report` | `evaluation_benchmark`: one lm-evaluation-harness run per benchmark (fingerprint: checkpoint content, task, few-shot, limit, decoding, harness version), then one report |
| `score:<run>:<task>`..., `rows`, `rows:<run>`..., `report` | `evaluation_compare`: the stages above per run, shared with the single-model recipes; the report compares every run with the base |
| `rows`, `score`, `report` | `evaluation_decision`: every question through the head without reasoning and after greedy reasoning; `items.jsonl` per question |
| `score:<task>`..., `report` | `evaluation_embedding`: one mteb run per task (fingerprint: checkpoint content, task, max_length, mteb version) |
| `clips`, `traces`, `report` | `evaluation_wakeword`: recordings and ambient WAVs found; the detector's score every 80 ms on all of them under each condition; the threshold sweep and the operating point |
| `data`, `train`, `validate` | `tts_piper`: recordings found and split; `piper.train fit` from the starting checkpoint, exported to `voice.onnx`; the voice reads the held-out texts and the fixed replies, Whisper transcribes them |
| `data`, `features`, `train`, `validate` | `wakeword`: recordings of both classes found and split; noisy windows as embeddings; the detector trained and exported; held-out streams scored every 80 ms |

| Recipe | Row keys | Metrics for `validation.min`/`max` |
|---|---|---|
| `llm_sft` | `messages` | `loss`, `perplexity` |
| `llm_instruction` | `instruction`, `input`?, `output` | `loss`, `perplexity` |
| `llm_dpo` | `prompt`, `chosen`, `rejected` | `accuracy` (chosen more likely than rejected), `margin` |
| `llm_grpo` | `prompt` + what the reward reads (`answer`) | `reward` (greedy completion) |
| `llm_decision_sft` | `state`, `questions` (Jev records) or `state`, `question`, `label` (one question per row) | `accuracy`, `nll`, `ece` |
| `llm_decision_cispo` | same | `accuracy`, `nll`, `ece`, `think_accuracy` (after reasoning) |
| `embedding_contrastive` | `query`, `positive`, `negative`? | `accuracy` (positive ranked first), `mrr` |
| `tts_piper` | `audio` (a WAV file), `text` | `wer` (Whisper on the voice), `reference_wer` (Whisper on the recordings), `duration_ratio`, `phrase_wer` (the worst-heard fixed reply), `utmos` and `dnsmos_sig`, `dnsmos_bak`, `dnsmos_ovrl` (how clean the voice sounds; `reference_` and `phrase_` for the recordings and the replies) |
| `wakeword` | `audio`, `text`? (a positive's must be the phrase) | `recall`, `false_accepts`, `scenario` (phrase then command: one firing, on the phrase) |

By default every weight is trained, in FP32. `training.adapter` switches any of these recipes to LoRA:

```yaml
training:
  adapter: {name: lora, r: 16, alpha: 32, dropout: 0.05}   # omit to train every weight
```

The base weights stay frozen and only the adapter trains, so the resume snapshot holds just the adapter. Which
layers are adapted comes from the architecture (`transformer`, `bi_encoder`: the attention projections; `hybrid`:
those plus the linear-attention projections) unless `targets` names others. The adapter is folded into the base
weights when the checkpoint is written, so a LoRA checkpoint and a fully trained one load the same way: a plain
`save_pretrained` directory at `output/<experiment>/train/checkpoint/` (and, once validation passed, at
`output/<experiment>/validate/checkpoint`).

### Decision recipes

A decision model answers `choice`, `noul` (yes/no) and `score` questions about a `state` with a probability per
option. The backbone gets a head (`model.head`) that reads the options, and the prompt marks state, question and
options with rare tokens of the Qwen tokenizer:

| Head | How it scores | Origin |
|---|---|---|
| `readout` | options get one-token codes (A, B, ...); a linear layer on the decide position scores the codes | Jev, jeff |
| `pointer` | a dot product between the decide position and each option's end position; no codes, any number of options | Jeeves |

`llm_decision_sft` trains the head with cross-entropy over the options. With `method.think_fraction: 0` that is a
one-pass decision model. Above 0, that share of the questions gets a reasoning chain sampled from the base model
before training and inserted as context, so the head learns to decide after reasoning as well as without it.

`llm_decision_cispo` continues from such a checkpoint (`model.init`): per question it samples `group_size`
reasoning chains, rewards each by the probability the head then gives the right option, and reinforces the chains
that beat their group's mean. The head's loss after reasoning and without reasoning is added, so the one-pass
answer keeps working.

Both keep `method.calibration` of the training records aside, fit one temperature on them when training ends and
store it with the head (`head.json` in the checkpoint); validation reports calibrated `accuracy`, `nll` and `ece`.
The training unit is the question: `training.batch_size` counts questions, however many a record holds, and CISPO
forwards `batch_size` x `group_size` sequences per step.

```yaml
model:
  head: pointer
  init: output/decision-pointer-qwen3.5-0.8b-<hash>/validate/checkpoint   # stage 2 starts where stage 1 ended
```

`model.init` works in every `llm_*` and `embedding_*` recipe: it is "start from this checkpoint of an earlier
experiment". Point it at `<experiment>/validate/checkpoint`: that link exists only when the earlier experiment
passed validation, whereas `train/checkpoint` is there even when it failed. The checkpoint counts by content in
the fingerprint, so rebuilding the earlier experiment rebuilds the one that continues it.

### Text-to-speech recipe (`tts_piper`)

`tts_piper` fine-tunes a [Piper](https://github.com/OHF-Voice/piper1-gpl) voice: a VITS model that reads espeak-ng
phonemes and runs anywhere as one ONNX file. Piper is GPL-3.0-or-later and pins its own torch, so it is not a
dependency of this package: it lives in `environment/piper/` (its own `pyproject.toml` and `uv.lock`), and both speech
recipes run it as a separate process through `PIPER_PYTHON`. Build it once, and again after pulling: the script is
safe to repeat.

```bash
environment/piper/setup.sh       # uv sync of the locked environment, plus the monotonic_align extension the wheel lacks
```

```yaml
model:
  quality: medium                  # must match the starting checkpoint; high sets Piper's larger decoder
  espeak_voice: en-gb-x-rp
  init:                            # a Piper .ckpt on the Hub (pinned), a local .ckpt, or <experiment>/validate/checkpoint
    name: jgkawell/jarvis
    revision: 37f8763122312665f091d1fc760abaf1f79b02cc
    file: en/en_GB/jarvis/medium/checkpoints/epoch=8338-step=2012650.ckpt
data:
  sources:
    - {name: delimited, path: recordings/metadata.csv, audio_dir: recordings/wavs}   # `<file>|<text>` lines
training: {epochs: 100, batch_size: 16, args: {model.learning_rate: 1.0e-4}}         # args: any other `piper.train fit` option
validation: {asr: {language: en}, max: {wer: 0.15}}
```

- **Rows.** `audio` and `text`. A relative `audio` path starts at the source's `audio_dir`, else at the directory of
  the source file, and an LJSpeech id without `.wav` finds its file. `delimited` reads a headerless `|` file such as
  an LJSpeech or Piper `metadata.csv`; `local_jsonl` reads the `rows.jsonl` that `data_speech` writes. The audio
  directories count in the `data` fingerprint, so re-recorded files rebuild the experiment.
- **Training.** The starting weights are copied in (`model.warmstart_ckpt`: every tensor whose shape matches, a fresh
  optimizer), so an old Piper checkpoint loads into the current trainer and `training.epochs` counts from zero.
  Piper's own validation split is off; the `validate` stage measures on this recipe's held-out rows instead. A rerun
  after a crash continues from Piper's last epoch snapshot (`run/lightning_logs/version_<n>/checkpoints/last.ckpt`);
  Piper turns cuDNN determinism off, so a GPU run is not bit-identical to an earlier one. `fit.log` holds Piper's
  output; a cancelled API job stops when the fit ends.
- **Checkpoint.** `<experiment>/validate/checkpoint/` holds `voice.onnx` and `voice.onnx.json` (a Piper voice:
  `piper -m voice.onnx`, Home Assistant's `/share/piper`) and `model.ckpt`, the start of a further fine-tune.
- **Validation.** The voice reads up to `validation.rows` held-out texts; Whisper (`validation.asr`, pinned) transcribes
  them and the original recordings. `reference_wer` is the floor `wer` can reach with that recognizer; numbers written
  as digits in one text and as words in the other count as errors.
- **Sound quality.** Whisper only says whether a voice is understood, not whether it sounds clean, so every clip is
  also scored by two predictors that need no reference recording, each from 1 to 5: `utmos` (naturalness, UTMOS22
  strong, MIT, as ONNX from `TigreGotico/utmos-onnx`) and DNSMOS P.835 (Microsoft DNS Challenge, CC BY 4.0):
  `dnsmos_sig` (distortion of the speech itself), `dnsmos_bak` (background noise) and `dnsmos_ovrl` (overall). The
  same metrics with `reference_` score the recordings, and with `phrase_` the kept replies; each take in `phrases.json`
  carries its own. Both models run on onnxruntime, the UTMOS one (411 MB) from the Hub cache and DNSMOS downloaded
  into `validate/models/` and checked against its pinned hash. Bound them like any metric, e.g.
  `min: {dnsmos_bak: 3.0}`.
- **Fixed replies.** `validation.phrases` (`name: text`) are the lines the agent plays as they are, such as the
  read-back before a task (`web_search: Searching Reference.`). Piper's noise makes every take of a text different,
  so each is spoken `validation.phrase_takes` times (default 3) and the take Whisper hears best becomes
  `<experiment>/validate/phrases/<name>.wav`; `phrases.json` lists every take and what was heard, and `phrase_wer` is
  the worst kept take, so a voice that garbles one reply fails validation. Changing a reply reruns only `validate`;
  a rerun keeps the takes it has, so the agent plays the file that passed.

Before training on a voice, read its terms. `jgkawell/jarvis` is MIT on its weights per its model card, but it imitates
a film character played by a real actor; MIT does not cover the voice, so treat it as personal and non-commercial.

### Wake word recipe (`wakeword`)

`wakeword` trains the detector that wakes a voice agent: a small model that listens to the microphone all the time
and fires when it hears one phrase. It follows [openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0):
audio becomes a melspectrogram and then a 96-number speech embedding every 80 ms through two frozen ONNX models
(pinned by hash, downloaded once), and a small `dnn` head scores the last 16 embeddings (2 s). Only the head is
trained, so a few hundred synthetic recordings are enough, and the exported file loads in openWakeWord's runtime.

```yaml
phrase: Jarvis                     # the wake word: the agent's name; what follows it is the command
data:
  positive: [{name: local_jsonl, path: output/wake-phrase-<hash>/select/rows.jsonl}]     # data/wake_phrase.yaml
  negative: [{name: local_jsonl, path: output/wake-negatives-<hash>/select/rows.jsonl},  # data/wake_negatives.yaml
             {name: local_jsonl, path: output/wake-nearmiss-<hash>/select/rows.jsonl}]   # data/wake_nearmiss.yaml
  background: []                   # directories of noise WAVs; empty: Gaussian noise
augment: {rounds: 4, followed: 2, noise: 200, snr: [5, 25]}            # followed: "Jarvis" + a command, as negatives
validation: {threshold: 0.5, min: {recall: 0.9, scenario: 0.9}, max: {false_accepts: 0.05}}
```

- **Phrase.** The wake word is the agent's name; the command after it ("Jarvis, wake up", "Jarvis, search the web
  for ...") is not part of it, and speech recognition hears it once the detector has fired. Positives say the phrase
  and nothing else (a positive row with a `text` must say it); negatives must never contain it, and the `data` stage
  stops on one that does. The hardest negatives are near misses ("Travis", "Davis", "Karen"), and the commands said
  alone. The model file is named after the phrase: `jarvis.onnx`.
- **Recordings.** `data_speech` with a multi-speaker voice (`voice.speakers`) and `select.per_text: null` makes them:
  the name read by hundreds of speakers, each checked by Whisper. Recordings of the people who will use the agent
  (`delimited` or `local_jsonl`) belong in `positive` too: synthetic voices are a start, not the target.
- **Windows.** Every training recording becomes `augment.rounds` 2-second windows over noise at a random SNR and gain,
  ending up to `augment.tail` seconds after the speech. Negatives are also cut short at random, so half a phrase is a
  negative; `augment.followed` windows per phrase put other speech after it, so the detector fires as the name ends
  and not again while the command that follows is spoken. All randomness comes from the seed and the recording.
- **Training.** Balanced batches (half positive), BCE, Adam, `training.steps` steps on the CPU, seeded: a rerun after a
  crash trains the same detector again.
- **Validation.** Held-out recordings in noise (`validation.snr`), scored every 80 ms as the runtime scores them, a
  firing counted at most once a second: `recall` and `false_accepts` per recording, and `scenario`, the agent's
  first moment: "Jarvis", a pause, then another held-out utterance (a command such as "Please search to the paper
  reference on the web", a near miss, or ordinary speech). It passes when the detector fires exactly once, while or
  just after the name is said.
- **Checkpoint.** `<experiment>/validate/checkpoint/` holds `jarvis.onnx`, the two feature models and
  `wakeword.json` (the phrase, the files, the feature models' hashes):

```python
from openwakeword.model import Model      # pip install openwakeword
detector = Model(wakeword_models=["checkpoint/jarvis.onnx"], inference_framework="onnx",
                 melspec_model_path="checkpoint/melspectrogram.onnx", embedding_model_path="checkpoint/embedding_model.onnx")
score = detector.predict(chunk)["jarvis"]   # chunk: 1280 samples, 16 kHz, int16; wake at >= the threshold
```

`false_accepts` is measured on the negatives you give, not on hours of real audio; openWakeWord's own models add a
large corpus of everyday speech and music for that. Add real background recordings and your own speech before
trusting the rate. openWakeWord's code is Apache-2.0 and its README puts its pre-trained models under CC BY-NC-SA 4.0;
the two feature models are derived from Google's Apache-2.0 `speech_embedding`, so treat a detector as
non-commercial until you have checked their terms.

### A voice agent from these recipes

The four configs below build what a voice agent says and hears; what it decides stays in the agent.

| Moment | What runs | Built by |
|---|---|---|
| User: "Jarvis, wake up" | the wake word detector fires as "Jarvis" ends, once | `wakeword` (`wakeword/jarvis.yaml`) |
| | speech to text hears the command after it ("wake up"), and the agent's model picks the command `wake_up` | the agent (an LLM recipe can tune it) |
| Agent: ready | plays the read-back of `wake_up`, `phrases/wake_up.wav` | `tts_piper` `validation.phrases` |
| User: "Please search to the paper reference on the web" | speech to text, then the agent's model picks `web_search` | the agent |
| Agent: "Searching Reference.", then works | plays `phrases/web_search.wav`, then runs the search | `tts_piper` `validation.phrases` |

`tts/piper_jarvis.yaml` holds nine read-backs: `wake_up` (At your service, sir.), `web_search` (Searching Reference.),
`horizontal_task` (Processing Long Task.), `remember_pattern` (Checking the Pattern.), `analysis` (Target.
Analysis.: a full stop so the voice pauses), `focus_mode_enter` (Activate Focus mode.), `focus_mode_quit` (Deactivate Focus mode.), `persona_karen`
(Change to KAREN.) and `persona_jarvis` (Change to JARVIS.). The voice's training set (`data/speech.yaml`) reads them
too. Which utterance means which command is the agent's to decide: no recipe here maps "enter the focus mode" to
`focus_mode_enter` yet.

## 03. Quick start

```bash
uv sync                                                    # Python 3.12
cp environment/.env.sample environment/.env                # HF_TOKEN for gated models; API_TOKEN, API_PATHS for a shared server

uv run llmclick recipes                                    # recipes, their stages and registry keys
uv run llmclick validate configs/llm/sft.yaml              # experiment key and stage plan
uv run llmclick run configs/llm/sft.yaml                   # the stages the config asks for

uv run llmclick serve --host 0.0.0.0 --port 8000           # API + Swagger at /docs (or: uvicorn core.api.app:app)
curl -X POST localhost:8000/api/jobs -H 'content-type: application/json' \
  -d '{"config_path": "configs/llm/sft.yaml"}'
```

| Config | What it builds |
|---|---|
| `llm/sft.yaml` | SFT of Qwen3.5-0.8B (hybrid) on conversations |
| `llm/sft_transformer.yaml` | the same recipe on Qwen3-0.6B (transformer) |
| `llm/instruction.yaml` | instruction tuning of Qwen3.5-0.8B |
| `llm/dpo.yaml` | DPO of Qwen3.5-0.8B on preference pairs |
| `llm/sft_lora.yaml` | the SFT recipe with a LoRA adapter instead of full fine-tuning |
| `llm/grpo.yaml` | GRPO of Qwen3.5-0.8B with the `exact_match` reward |
| `llm/grpo_gsm8k.yaml` | GRPO on GSM8K word problems from the Hub with the `last_number` reward |
| `llm/decision_readout.yaml` | one-pass decision model: full fine-tuning, readout head (Jev, jeff) |
| `llm/decision_pointer.yaml` | reasoning decision model, stage 1: LoRA, pointer head, reasoning chains as context (Jeeves SFT) |
| `llm/decision_cispo.yaml` | reasoning decision model, stage 2: CISPO from the stage-1 checkpoint |
| `embedding/contrastive.yaml` | contrastive learning of Qwen3-Embedding-0.6B |
| `evaluation/custom.yaml` | the `llm/sft.yaml` experiment's validated checkpoint scored on the sample rows |
| `evaluation/benchmark.yaml` | the same checkpoint on the `small_general` preset plus KMMLU, 100 items each (needs `uv sync --extra eval`) |
| `evaluation/compare.yaml` | base model against the trained checkpoint on the preset and the sample rows, with a four-axis chart |
| `evaluation/decision.yaml` | the `llm/decision_pointer.yaml` checkpoint on the JevBench tiers, without and with reasoning |
| `evaluation/embedding.yaml` | the `embedding/contrastive.yaml` checkpoint on three MTEB tasks (needs `uv sync --extra eval`) |
| `data/synthetic.yaml` | Korean conversation rows for `llm_sft` from four topic seeds through a local Ollama model, evolved, judged, deduplicated |
| `data/speech.yaml` | a Piper training set: assistant lines written by a local Ollama model, read by the `jgkawell/jarvis` voice, checked by Whisper (needs `environment/piper/setup.sh`) |
| `tts/piper_jarvis.yaml` | fine-tuning from the `jgkawell/jarvis` medium checkpoint on that set, with the agent's nine read-backs; replace `<hash>` with the `data/speech.yaml` experiment's |
| `data/wake_phrase.yaml` | "Jarvis." (statement, question, call) read by 300 speakers of the `libritts_r` Piper voice, kept if Whisper hears it exactly |
| `data/wake_negatives.yaml` | near misses of the name, the commands said alone, and ordinary lines, read by 40 speakers of the same voice |
| `data/wake_nearmiss.yaml` | 20 near misses of the name ("Travis", "Jars", "Jarrett", ...) by 120 more speakers, kept without a listening check |
| `wakeword/jarvis.yaml` | the "Jarvis" wake word detector from those three sets; replace each `<hash>` |

The `llm/` and `embedding/` configs ship as pilots: sample rows from `samples/` and `max_steps: 4`. Point
`data.sources` at your rows and remove `max_steps` for a real run.

Docker: `IMAGE_TAG=$(git rev-parse --short HEAD) docker compose -f environment/docker-compose.yml up -d --build`.

Installed as a package (`uv build`, then `pip install dist/llmclick-*.whl`), `llmclick` runs from any directory:
config paths, `output_dir` and `environment/.env` are relative to the working directory, and `LLMCLICK_ENV` names a
different settings file.

### Apple silicon

Training and validation run on MPS; `device` in the config (or auto-detection: cuda, then mps, then cpu) decides.
Weights are trained in FP32: fully fine-tuning Qwen3.5-0.8B needs roughly 13 GB of unified memory at the shipped
batch sizes, and `training.adapter` (LoRA) cuts that to little more than the weights themselves.

## 04. Extending

Each family is a model catalog crossed with training methods; the two grow independently.

```
modeling
├── llm
│   ├── Model Catalog      transformer, hybrid            <- add an architecture
│   │   └── heads          readout, pointer               <- add a decision head
│   └── Training Method    sft, instruction, dpo, grpo    <- add a method
│       └── decision       sft, cispo
├── embedding
│   ├── Model Catalog      bi_encoder
│   └── Training Method    contrastive
├── tts                    one recipe, trained by the Piper program; `model.quality` picks its size
└── wakeword               one recipe: openWakeWord's dnn head on its frozen features
```

### A new architecture in a catalog

Subclass the family's backbone and register it; every recipe of the family can then name it in `model.architecture`.

```python
@LLM_BACKBONES.register("moe")                  # modeling/llm/models/moe.py; import it in models/__init__.py
class MoEBackbone(LLMBackbone):
    loader = AutoModelForCausalLM               # the transformers class that builds it
    frozen = ("router",)                        # parameter-name fragments training must not update
    adapter_targets = ("q_proj", "v_proj")      # layers LoRA adapts by default
    markers = Markers(...)                      # only for decision recipes: seven single tokens of its tokenizer
```

An embedding architecture (for example a cross-encoder) subclasses `EmbeddingBackbone` the same way and overrides
what differs, such as `pool`.

### A new decision head

A head owns the prompt it reads and turns hidden states into one score per option. The tokens that mark state,
question and options come from the architecture (`LLMBackbone.markers`), so a head works on any backbone that
defines them.

```python
@HEADS.register("my_head")                      # modeling/llm/models/heads/my_head.py; import it in heads/__init__.py
class MyHead(DecisionHead):
    def option_block(self, options): ...        # how the options appear in the prompt
    def scores(self, hidden, layouts): ...      # (rows, options) scores from (rows, tokens, hidden) states
```

### A new training method in a family

A method says what a row must contain, how a batch becomes a loss and how the result is measured. Then one config
class and one registered pipeline make it a recipe.

```python
class MyMethod(LLMMethod):                      # modeling/llm/methods/my_method.py
    class Config(BaseModel):                    # the YAML's `method:` section
        strength: float = 1.0

    def check(self, row): ...                   # raise ValueError for a row this method cannot use
    def loss(self, backbone, rows): ...         # differentiable scalar for one batch
    def evaluate(self, backbone, rows, batch_size, progress): ...   # metrics the validation bounds can name

class MyConfig(LLMConfig):                      # modeling/llm/config.py
    method_class = MyMethod
    method: MyMethod.Config = Field(default_factory=MyMethod.Config)

@recipe                                         # modeling/llm/pipeline.py
class MyPipeline(LLMPipeline):
    kind = "llm_my_method"
    config_class = MyConfig
```

`prepare(backbone, rows, workdir)` runs once before training with the untouched base weights (DPO caches its
reference margins there). Helpers shared by LLM methods live in `modeling/llm/methods/base.py`.

### A new family

A package with `models/`, `methods/`, a `TuningConfig` subclass that sets `backbones` to its catalog, and a
`TuningPipeline` subclass; import it in `modeling/__init__.py`. The data, train and validate stages and the training
loop are inherited.

### Reusing sampled groups (`llm_grpo`, `llm_decision_cispo`)

Sampling is the expensive part of both methods. `method.iterations` takes that many optimizer steps on each sampled
group before sampling again. From the second step on the policy has moved away from the one that produced the
samples, and `method.clip` bounds how far: GRPO clips the policy ratio to 1 +- clip (PPO's surrogate), CISPO caps the
importance weight of each reasoning token at 1 + clip. `llm_grpo` also takes `method.beta`, a KL penalty towards
the base model; it needs `training.adapter`, because the base model is then the same weights with the adapter
switched off and no second model is loaded.

```yaml
method: {group_size: 4, iterations: 2, clip: 0.2, beta: 0.05}
training: {adapter: {name: lora, r: 16, alpha: 32}}
```

### A reward for `llm_grpo`

```python
@REWARDS.register("json_valid")                 # modeling/llm/methods/grpo.py
def json_valid(completion: str, row: dict, params: dict) -> float: ...
```

### A recipe with its own stages

When training does not fit the shared loop (for example an external trainer run as a child process), a recipe
brings its own config class, stages and pipeline class; users only see a new value for `pipeline.recipe`.

```python
class MyConfig(ModelingConfig):                 # modeling/<recipe>/config.py
    model: MyModelConfig

class MyTrain(TrainStage[MyConfig]):            # modeling/<recipe>/stages.py
    sections = ("seed", "model")                # config sections that decide the result
    def train(self, workdir, inputs):           # write into workdir; must be safe to run again after a crash
        ...
        return {"checkpoint": str(workdir / "model")}

class MyValidate(ValidateStage[MyConfig]):
    def measure(self, workdir, inputs):         # metrics on held-out data; bounds and report are handled for you
        return {"accuracy": ...}

@recipe                                         # modeling/<recipe>/pipeline.py; import it in modeling/__init__.py
class MyPipeline(ModelingPipeline[MyConfig]):
    kind = "my_recipe"
    config_class = MyConfig
    stage_classes = (MyTrain, MyValidate)       # execution order; a stage may only require earlier ones
```

Inside a stage, `self.progress.update(done, total, note)` drives the progress bar.

### Building blocks a new recipe reuses

A data, evaluation or modeling recipe is safe to rerun because its stages write through these. Use them rather than
writing the step again, so the guarantee holds in one place:

| Need | Use | Where |
|---|---|---|
| Write a file whole or not at all | `with atomic_path(path) as temporary:` | `core/utils/files.py` |
| Build a checkpoint directory whole or not at all | `with staged_directory(workdir / "checkpoint") as staging:` | `core/utils/files.py` |
| Rows between stages | `read_jsonl`, `write_jsonl` (atomic); the data channel's `read_rows` also reads a missing file as empty | `core/utils/files.py`, `data/rows.py` |
| Per-item results that survive a crash | `cached(directory, key)`, `keep(directory, key, row)` under a natural `key(...)` | `data/rows.py` |
| Hold out validation rows | `held_out_positions(count, seed, share)`: a row's side depends only on the seed and its position | `modeling/stages.py` |
| Load a model for one block | `with backbone.loaded(device, checkpoint):` releases it afterwards, also on failure | `modeling/tuning/backbone.py` |
| A `name:` key in the config | a `Registry`; in the config validator `REGISTRY.check(*names)` gives the standard error | `core/registry.py` |
| Local files a config reads | `local_paths(entries, keys)` in `paths()`, so the API can confine them | `modeling/config.py` |
| Recordings named by rows | `audio.recordings(source)`, `audio.signatures(sources)` for the fingerprint | `modeling/tts/audio.py` |

## 05. Tests

```bash
uv run pytest tests/core tests/modeling  # runner, progress, validation gate, API, every recipe on a tiny random model (wakeword: a fake front end)
uv run pytest tests/data                # data recipes with fake teachers, a fake Piper and a fake Whisper
uv run ruff check . && uv run mypy      # what CI runs on every pull request, besides the tests
```

Licence: MIT. Models and datasets a config names keep their own licences.

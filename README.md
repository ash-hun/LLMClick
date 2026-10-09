# LLMClick

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
| Evaluation | `evaluation_custom` | nothing: scores a checkpoint of any experiment above on any rows, with that recipe's metrics | the experiment's |
| Evaluation | `evaluation_benchmark` | nothing: scores a checkpoint on public benchmarks (lm-evaluation-harness), one cached stage per benchmark | the experiment's (LLM) |
| Evaluation | `evaluation_compare` | nothing: several checkpoints (the base model among them) on the same benchmarks and rows; table, deltas with intervals, polygon chart | the experiments' |

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
│   └── embedding/                #   family: text embedding models
│       ├── models/               #     Model Catalog: base.py (EmbeddingBackbone), bi_encoder.py
│       ├── methods/              #     Training Method: contrastive.py
│       └── config.py pipeline.py
├── data/                         # Data channel (planned): README only
├── evaluation/                   # Evaluation channel: config -> scores of a trained model, report.json
│   ├── source.py                 #   SourceExperiment: an experiment's config read back, its checkpoint located
│   ├── stages.py                 #   rows, score, report (evaluation_custom)
│   ├── benchmark.py              #   presets, one score stage per benchmark over lm-evaluation-harness (evaluation_benchmark)
│   ├── compare.py                #   runs, verdicts against the base, polygon SVG, Markdown summary (evaluation_compare)
│   └── config.py pipeline.py     #   (README: design and milestones)
├── tests/core/  tests/modeling/
├── docs/                         # api-spec.md, screen-spec.md
├── environment/                  # Dockerfile, docker-compose.yml, .env.sample
└── output/
    ├── _stages/<stage>-<fingerprint>/   # what each stage built; shared by every experiment with the same inputs
    └── <name>-<hash>/                   # one experiment: config.yaml, manifest.json, pipeline.log, links to its stages
```

`evaluation/` scores what `modeling/` built: `evaluation_custom` measures any checkpoint of an experiment (`validate`,
`train`, `base`, or a directory) on any rows with the metrics of the recipe that trained it; `evaluation_benchmark`
scores it on public benchmarks through lm-evaluation-harness (`uv sync --extra eval`), one cached stage per
benchmark, with presets for the small-model panel and Korean; `evaluation_compare` puts several checkpoints, the
base model among them, through the same benchmarks and rows and reports the differences with their intervals, a
polygon chart and a Markdown summary. All write `report.json`; decision and embedding recipes follow
(`evaluation/README.md` has the design and milestones). `data/` (training and synthetic data pipelines) is planned and has a README only. A channel is a
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

| Recipe | Row keys | Metrics for `validation.min`/`max` |
|---|---|---|
| `llm_sft` | `messages` | `loss`, `perplexity` |
| `llm_instruction` | `instruction`, `input`?, `output` | `loss`, `perplexity` |
| `llm_dpo` | `prompt`, `chosen`, `rejected` | `accuracy` (chosen more likely than rejected), `margin` |
| `llm_grpo` | `prompt` + what the reward reads (`answer`) | `reward` (greedy completion) |
| `llm_decision_sft` | `state`, `questions` (Jev records) or `state`, `question`, `label` (one question per row) | `accuracy`, `nll`, `ece` |
| `llm_decision_cispo` | same | `accuracy`, `nll`, `ece`, `think_accuracy` (after reasoning) |
| `embedding_contrastive` | `query`, `positive`, `negative`? | `accuracy` (positive ranked first), `mrr` |

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
└── embedding
    ├── Model Catalog      bi_encoder
    └── Training Method    contrastive
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

## 05. Tests

```bash
uv run pytest tests/core tests/modeling  # runner, progress, validation gate, API, every recipe on a tiny random model
uv run ruff check . && uv run mypy      # what CI runs on every pull request, besides the tests
```

Licence: MIT. Models and datasets a config names keep their own licences.

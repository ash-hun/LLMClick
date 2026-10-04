# LLMClick

Config-driven custom model building. One YAML describes one custom model; `pipeline.recipe` picks the recipe that
builds it and `pipeline.stages` picks the steps to run. Every recipe runs on the same pipeline runner, which skips
what is already built, shows progress, and never hands on a trained model that has not passed validation.

The first recipe is `jev`: one-pass decision models (describe the situation and the options, get calibrated
probabilities over the options in one forward pass), trained with [jeff](https://github.com/firelex/jeff) (itself a
fork of [AutoJev](https://github.com/denis-pplx/autojev)), vendored unchanged under `jeff/`.

## 01. Project structure

```
LLMClick/
├── main.py                       # FastAPI entry point
├── config.py                     # process settings (environment/.env): HF_TOKEN, TEACHER_URL ...
├── configs/                      # one YAML = one custom model
├── core/                         # the framework, shared by every channel
│   ├── stage.py                  #   Stage: one fingerprinted, rerunnable unit of work
│   ├── pipeline.py               #   Pipeline: runs a recipe's stages; build()/load() turn a config into one
│   ├── progress.py               #   Progress, BarProgress (terminal), StateProgress (API jobs)
│   ├── registry.py               #   Registry, RECIPES, CHANNELS
│   ├── config/                   #   schema.py (BaseConfig), experiment.py (directories, manifest)
│   ├── cli.py                    #   llmclick run|validate|recipes
│   ├── api/                      #   routers: config_channel, job_channel, system_channel; store.py
│   └── utils/                    #   files (hash, lock), proc (child processes), device
├── modeling/                     # Modeling channel: config -> trained, validated model
│   ├── config.py                 #   ModelingConfig: tracker, validation bounds
│   ├── stages.py                 #   TrainStage, ValidateStage (templates every recipe fills in)
│   ├── pipeline.py               #   ModelingPipeline: training always brings validation
│   ├── tracker.py                #   wandb replay
│   └── jev/                      #   recipe `jev`
│       ├── config.py             #     JevConfig
│       ├── stages.py             #     data, benchmarks, synthetic, mix, train, validate, evaluate
│       ├── pipeline.py           #     JevPipeline (registered as `jev`)
│       ├── registry.py           #     BUILDERS, CONVERTERS, BACKBONES, TEACHERS, BENCHMARKS
│       └── data/ model/ tuning/ evaluation/
├── jeff/                         # vendored jeff package (MIT); local patches listed in jeff/PATCHES.md
├── tests/core/  tests/modeling/  tests/jeff/
├── docs/                         # api-spec.md, screen-spec.md, data-sources.md
├── environment/                  # Dockerfile, docker-compose.yml, .env.sample
└── output/
    ├── _stages/<stage>-<fingerprint>/   # what each stage built; shared by every experiment with the same inputs
    └── <name>-<hash>/                   # one experiment: config.yaml, manifest.json, pipeline.log, links to its stages
```

Two more channels are planned next to `modeling/`: Data (training and synthetic data pipelines) and Evaluation
(custom and benchmark evaluation with reports). A channel is a package that registers recipes; it joins by adding
its name to `CHANNELS` in `core/registry.py` and reuses `Stage`, `Pipeline`, `Progress` and `Experiment` as they are.

## 02. How a run works

```yaml
pipeline:
  recipe: jev                       # which pipeline builds this model
  name: my-model
  stages: [data, benchmarks, mix]   # optional; default all. Stages they read are added, train always brings validate
```

- **Fingerprints.** A stage's fingerprint is its own config inputs (plus the content of local files it reads) and
  the fingerprints of the stages it reads. Its directory is `output/_stages/<stage>-<fingerprint>/`, so two configs
  that differ only in training settings share the data stages, and a rerun of the same config builds nothing.
- **Reruns.** A stage is recorded only when it finishes. A crashed run is resumed by running the same command
  again; training continues from `checkpoints/resume.pt`.
- **Locks.** A stage directory is locked while it is built, so a CLI run and an API job asking for the same stage
  never build it twice.
- **Identity.** The experiment directory is `<name>-<hash>`; `stages`, `output_dir` and `tracker` are not part of
  the hash, because they say how to run, not what to build.
- **Progress.** The CLI shows two bars (stages, and steps inside the running stage); API jobs report the same
  state in `progress`. Output of child processes goes to log files in the stage directory.
- **Validation.** In the Modeling channel `validate` follows `train` and cannot be left out. It measures the
  trained model on data training never saw and checks `validation.min` / `validation.max`; a miss stops the
  pipeline, and `evaluate` reads the checkpoint from `validate`, never from `train`.

### Stages of `jev`

| Stage | What it does | jeff parts used |
|---|---|---|
| `data` | run every `data.builders[*]`, concatenate, carve `dev`/`temperature`/`validation` folds by family | `jeff.extra`, `jeff.probability`, `jeff.data.choose/validate/write_rows` |
| `benchmarks` | freeze every `evaluation.benchmarks[*]` | `jeff.panel`, `jeff.jevbench` |
| `synthetic` | teacher-written questions with verify/review/repair (only if `data.synthetic.enabled`) | `jeff.materials`, `jeff.grounded`, `jeff.generate` |
| `mix` | leak filter against the benchmarks and the validation fold, panel layouts, escape options, hijack attempts, size caps | `jeff.mix`, `jeff.layout`, `jeff.escape`, `jeff.adversarial` |
| `train` | full-weight SFT, checkpoint selection on dev NLL, fitted temperature | `jeff.train` |
| `validate` | accuracy, ECE, Brier, NLL on the validation fold against `validation.min`/`max` | `jeff.evaluate` |
| `evaluate` | accuracy, ECE, Brier, NLL per benchmark | `jeff.evaluate` |

## 03. Quick start

```bash
uv sync                                                    # Python 3.12, pins from jeff
cp environment/.env.sample environment/.env                # HF_TOKEN, TEACHER_URL ...

uv run llmclick validate configs/jeff_public_only.yaml     # experiment key and stage plan
uv run llmclick run configs/jeff_public_only.yaml          # the stages the config asks for
uv run llmclick recipes                                    # recipes, their stages and registry keys

uv run uvicorn main:app --host 0.0.0.0 --port 8000         # API + Swagger at /docs
curl -X POST localhost:8000/api/jobs -H 'content-type: application/json' \
  -d '{"config_path": "configs/jeff_public_only.yaml"}'
```

| Config | What it builds |
|---|---|
| `jeff_public_only.yaml` | jeff's public-only arm (no teacher) |
| `jeff_combined.yaml` | public + synthetic data from a local teacher |
| `custom_dataset.yaml` | your own HF/local data through YAML-only converters |
| `jeff_0.8b_mac.yaml` | the public-only recipe on Apple silicon (MPS) |
| `mac_smoke.yaml` | minutes-long end-to-end pilot on a Mac |
| `eval.yaml` | scores an existing checkpoint |

Docker: `IMAGE_TAG=$(git rev-parse --short HEAD) docker compose -f environment/docker-compose.yml up -d --build`.

### Apple silicon

Training and evaluation run on MPS. `device` in the config (or auto-detection: cuda, then mps, then cpu) is passed to
jeff as `JEFF_DEVICE`; `jeff/PATCHES.md` lists the small changes that made jeff's trainer device-agnostic.

```bash
uv run llmclick run configs/mac_smoke.yaml      # ~minutes: 0.8B, 4 updates, code-built data, proves the path
uv run llmclick run configs/jeff_0.8b_mac.yaml  # the real public-only recipe; many hours on an M-series
```

Memory: full-weight 0.8B in bf16 plus FP32 master weights and Adam moments in host memory needs roughly 12 GB of
unified memory at `batch_size: 8`, `token_budget: 4096`; lower both if the run is killed. Gemma and the 2B student
have not been tried on MPS.

## 04. Extending

### A registry entry of `jev`

Everything external is a `name:` key in the YAML (`uv run llmclick recipes` lists them):

| Registry | Keys | Extend |
|---|---|---|
| builder | `local_jsonl`, `huggingface`, `jeff_extra`, `jeff_probability` | `modeling/jev/data/builders.py` |
| converter | `example`, `classification`, `boolean` | `modeling/jev/data/converters.py` |
| backbone | `qwen3_5`, `gemma4`, `modernbert` | `modeling/jev/model/backbones.py` |
| teacher | `openai_compatible` | `modeling/jev/data/synthetic.py` |
| benchmark | `local_jsonl`, `huggingface`, `jeff_panel`, `jeff_jevbench_hard`, `jeff_probability` | `modeling/jev/evaluation/benchmarks.py` |

A Hugging Face classification set needs no code: see `configs/custom_dataset.yaml`, which maps `premise`/`hypothesis`
and a label column to jeff's `choice` question through the `classification` converter. Pin `revision` to a commit.

```python
from modeling.jev.registry import BENCHMARKS

@BENCHMARKS.register("kobest_boolq")
def kobest_boolq(params: dict, out: Path, seed: int) -> Path:
    ...  # write Example rows to out/rows.jsonl and return the path; never overwrite a different file
```

### A new recipe

A recipe is a config class, its stages and a pipeline class; users only see a new value for `pipeline.recipe`.

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

## 05. Data contract

All stages exchange jeff `Example` rows (`jeff/types.py`): `id`, `suite`, `family`, `state`, `question`
(`choice` with `criteria` or `noul`), `label`, `target`, `source`. `family` groups rows that must stay in one fold.

## 06. Tests

```bash
uv run pytest tests/core tests/modeling  # runner, progress, validation gate, API, jev config and idempotent data stages
uv run pytest tests/jeff -m "not slow"   # jeff's own tests on the vendored code
```

Licences: LLMClick code MIT; `jeff/` MIT (see `jeff/LICENSE`); training data keep their own licences
(`docs/data-sources.md`).

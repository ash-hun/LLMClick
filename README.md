# LLMClick

Config-driven training pipeline for Jev-style decision models: describe the situation and the options, get calibrated
probabilities over the options in one forward pass. The recipe is [jeff](https://github.com/firelex/jeff) (itself a
fork of [AutoJev](https://github.com/denis-pplx/autojev)), vendored unchanged under `jeff/`. LLMClick adds what
[PolyBed-Pipeline](reference/PolyBed-Pipeline) has for embeddings: one YAML per experiment, string-keyed registries
for every external dependency, a CLI and a FastAPI job server.

## 01. Project structure

```
LLMClick/
├── main.py                       # FastAPI entry point
├── config.py                     # process settings (environment/.env): HF_TOKEN, TEACHER_URL, JEFF_DEVICE ...
├── configs/                      # one YAML = one experiment
│   ├── jeff_public_only.yaml     #   reproduce jeff's public-only arm (no teacher)
│   ├── jeff_combined.yaml        #   public + synthetic data from a local teacher
│   ├── custom_dataset.yaml       #   your own HF/local data through YAML-only converters
│   ├── jeff_0.8b_mac.yaml        #   the public-only recipe on Apple silicon (MPS)
│   ├── mac_smoke.yaml            #   minutes-long end-to-end pilot on a Mac
│   └── eval.yaml                 #   score an existing checkpoint
├── core/
│   ├── registry.py               # BUILDERS, CONVERTERS, BACKBONES, TEACHERS, BENCHMARKS
│   ├── pipeline.py               # stage orchestration + manifest fingerprints
│   ├── run.py                    # CLI: llmclick run|validate|registries
│   ├── config/                   # schema.py (Pydantic), experiment.py (directory = name + config hash)
│   ├── api/                      # routers: config_channel, job_channel, system_channel; store.py
│   └── modules/
│       ├── data/                 # builders, converters, materialize, folds, synthetic, mix
│       ├── model/backbones.py    # backbone registry -> jeff-train arguments
│       ├── tuning/               # trainer (jeff.train subprocess), tracker (wandb replay)
│       └── evaluation/           # benchmarks registry, evaluator (jeff.evaluate subprocess)
├── jeff/                         # vendored jeff package (MIT); local patches listed in jeff/PATCHES.md
├── tests/core/  tests/jeff/      # ours / jeff's own
├── docs/                         # api-spec.md, screen-spec.md, data-sources.md
├── environment/                  # Dockerfile, docker-compose.yml, .env.sample
└── output/<name>-<hash>/         # experiments: config.yaml, manifest.json, data/, runs/, checkpoints/, eval/
```

## 02. Pipeline

| Stage | What it does | jeff parts used |
|---|---|---|
| `data` | run every `data.builders[*]`, concatenate, carve `dev`/`temperature` folds by family | `jeff.extra`, `jeff.probability`, `jeff.data.choose/validate/write_rows` |
| `benchmarks` | freeze every `evaluation.benchmarks[*]` (also protected by the leak filter) | `jeff.panel`, `jeff.jevbench` |
| `synthetic` | teacher-written questions with verify/review/repair (only if `data.synthetic.enabled`) | `jeff.materials`, `jeff.grounded`, `jeff.generate` |
| `mix` | leak filter against the benchmarks, panel layouts, escape options, hijack attempts, size caps | `jeff.mix`, `jeff.layout`, `jeff.escape`, `jeff.adversarial` |
| `train` | full-weight SFT, checkpoint selection on dev NLL, fitted temperature | `jeff.train` |
| `evaluate` | accuracy, ECE, Brier, NLL per benchmark | `jeff.evaluate` |

Every stage writes its outputs and a fingerprint (its config sections + upstream fingerprints) into
`output/<experiment>/manifest.json`. Rerunning the same config skips finished stages; a crashed training run resumes
from `checkpoints/resume.pt`. The experiment directory is named by the config hash, so a changed config is a new
experiment and an unchanged one is the same.

## 03. Quick start

```bash
uv sync                                                    # Python 3.12, pins from jeff
cp environment/.env.sample environment/.env                # HF_TOKEN, TEACHER_URL ...

uv run llmclick validate configs/jeff_public_only.yaml     # prints the experiment key
uv run llmclick run configs/jeff_public_only.yaml          # all stages
uv run llmclick run configs/jeff_public_only.yaml --stages data,benchmarks,mix
uv run llmclick registries                                 # what the YAML keys can be

uv run uvicorn main:app --host 0.0.0.0 --port 8000         # API + Swagger at /docs
curl -X POST localhost:8000/api/jobs -H 'content-type: application/json' \
  -d '{"config_path": "configs/jeff_public_only.yaml"}'
```

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

## 04. Swapping external dependencies

Everything external is a `name:` key in the YAML. Registered keys (`uv run llmclick registries`):

| Registry | Keys | Extend |
|---|---|---|
| builder | `local_jsonl`, `huggingface`, `jeff_extra`, `jeff_probability` | `core/modules/data/builders.py` |
| converter | `example`, `classification`, `boolean` | `core/modules/data/converters.py` |
| backbone | `qwen3_5`, `gemma4`, `modernbert` | `core/modules/model/backbones.py` |
| teacher | `openai_compatible` | `core/modules/data/synthetic.py` |
| benchmark | `local_jsonl`, `huggingface`, `jeff_panel`, `jeff_jevbench_hard` | `core/modules/evaluation/benchmarks.py` |

A Hugging Face classification set needs no code: see `configs/custom_dataset.yaml`, which maps `premise`/`hypothesis`
and a label column to jeff's `choice` question through the `classification` converter. Pin `revision` to a commit.

Adding a new one:

```python
from core.registry import BENCHMARKS

@BENCHMARKS.register("kobest_boolq")
def kobest_boolq(params: dict, out: Path, seed: int) -> Path:
    ...  # write Example rows to out/rows.jsonl and return the path; never overwrite a different file
```

## 05. Data contract

All stages exchange jeff `Example` rows (`jeff/types.py`): `id`, `suite`, `family`, `state`, `question`
(`choice` with `criteria` or `noul`), `label`, `target`, `source`. `family` groups rows that must stay in one fold.

## 06. Tests

```bash
uv run pytest tests/core                # schema, registries, folds, converters, API, idempotent data/benchmarks/mix
uv run pytest tests/jeff -m "not slow"  # jeff's own tests on the vendored code
```

Licences: LLMClick code MIT; `jeff/` MIT (see `jeff/LICENSE`); training data keep their own licences
(`docs/data-sources.md`).

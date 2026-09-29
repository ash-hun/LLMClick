# Local patches to the vendored jeff package

`jeff/` is [firelex/jeff](https://github.com/firelex/jeff) `src/jeff` copied as is, except the lines marked
`PATCH(llmclick)`. Re-apply them when replacing the directory with a newer jeff.

| File | Change | Why |
|---|---|---|
| `train.py` | `git rev-parse HEAD` failure → `"unknown"` | LLMClick may have no commits, and an installed wheel has no `.git` |
| `train.py` | `uv.lock` hashed from `package.parent`, `"absent"` if missing | the package sits at the repository root, not `src/jeff` |
| `train.py` | model loaded with `device=device_from_environment()` | `JEFF_DEVICE=mps` must reach training, not only serving |
| `train.py` | `torch.cuda.synchronize()` / `max_memory_allocated()` → `synchronize()` / `peak_memory_gb()` | those raise on a build without CUDA (Apple silicon) |
| `optim.py` | barrier after non-blocking copies on MPS too | the shared staging buffer is reused per parameter |
| `evaluate.py` | model loaded with `device=device_from_environment()` | same as training |

Everything else, including the tests under `tests/jeff/`, is unchanged apart from one import path in
`tests/jeff/test_generate.py` (`tests.test_materials` → `tests.jeff.test_materials`).

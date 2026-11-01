#!/bin/sh
# llama.cpp's HF-to-GGUF converter at a pinned release, for llm_quantize's gguf scheme (LLAMA_CPP_DIR's default path).
# Only the Python converter is used, run with the app's environment (`uv sync --extra quantize`); nothing is compiled.
# Safe to run again: the source is fetched only when the pinned commit is not already in place.
set -eu
cd "$(dirname "$0")"
# the app's interpreter (3.12): tarfile's extraction filter needs it, and the converter runs with it anyway
uv run --project ../.. --no-sync python - <<'PY'
import hashlib
import shutil
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

# release v0.7.0 (2026-10-15)
COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
SHA256 = "1d4a987d9bd048f6e93107e838026438df70fbde237fc0c69c63d600512b4bd9"
URL = f"https://github.com/ggml-org/llama.cpp/archive/{COMMIT}.tar.gz"

target = Path("llama.cpp")
if (target / "COMMIT").is_file() and (target / "COMMIT").read_text().strip() == COMMIT:
    sys.exit(0)
archive = urlopen(URL, timeout=300).read()
if hashlib.sha256(archive).hexdigest() != SHA256:
    sys.exit(f"{URL} does not match the pinned hash")
staging = Path("llama.cpp.tmp")
shutil.rmtree(staging, ignore_errors=True)
with tarfile.open(fileobj=BytesIO(archive)) as bundle:
    bundle.extractall(staging, filter="data")
shutil.rmtree(target, ignore_errors=True)
(staging / f"llama.cpp-{COMMIT}").rename(target)
shutil.rmtree(staging)
(target / "COMMIT").write_text(COMMIT + "\n")
PY

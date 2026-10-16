#!/bin/sh
# The Piper environment: the locked packages, plus monotonic_align, which the piper-tts wheel ships without its source.
# Safe to run again: uv converges the packages and the extension is built only when it is missing.
set -eu
cd "$(dirname "$0")"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"  # the app image points this at the app's environment
uv sync --frozen
.venv/bin/python - <<'PY'
import hashlib
import subprocess
import sys
import importlib.util
from pathlib import Path
from urllib.request import urlopen

# core.pyx of the commit tagged v1.8.0, the piper-tts version pyproject.toml pins
URL = "https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/639388b6317fc4731e91d53da42aea68fd4166ff/src/piper/train/vits/monotonic_align/core.pyx"
SHA256 = "8640b303683823a4a1259179547ef476999b1cbb2e46ff656b970763cfbc1157"

spec = importlib.util.find_spec("piper")
target = Path(spec.submodule_search_locations[0]) / "train" / "vits" / "monotonic_align" / "monotonic_align"
if not list(target.glob("core*.so")):
    target.mkdir(parents=True, exist_ok=True)
    (target / "__init__.py").touch()
    source = urlopen(URL, timeout=60).read()
    if hashlib.sha256(source).hexdigest() != SHA256:
        sys.exit(f"core.pyx from {URL} does not match the pinned hash")
    (target / "core.pyx").write_bytes(source)
    subprocess.run([sys.executable, "-m", "Cython.Build.Cythonize", "-i", "-3", "core.pyx"], cwd=target, check=True)
PY
.venv/bin/python -c "from piper.train.vits.monotonic_align import maximum_path"
echo "piper environment ready: $PWD/.venv/bin/python"

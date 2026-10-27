#!/bin/sh
# The Qwen3-TTS environment: the locked packages. Apple silicon only, since MLX runs on Metal.
# Safe to run again: uv converges the packages.
set -eu
cd "$(dirname "$0")"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync --frozen
.venv/bin/python -c "import mlx.core, mlx_audio.tts.utils"
echo "qwen environment ready: $PWD/.venv/bin/python"

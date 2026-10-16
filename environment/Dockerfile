FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

# build-essential: environment/piper/setup.sh compiles Piper's monotonic_align extension
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv PATH="/app/.venv/bin:$PATH"

# dependency layer: only the lock and project metadata
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# the Piper program (tts_piper, data_speech) in its own environment, at PIPER_PYTHON's default path
COPY environment/piper/pyproject.toml environment/piper/uv.lock environment/piper/setup.sh ./environment/piper/
RUN environment/piper/setup.sh

COPY main.py ./
COPY core/ ./core/
COPY modeling/ ./modeling/
COPY evaluation/ ./evaluation/
COPY data/ ./data/
COPY samples/ ./samples/
COPY configs/ ./configs/
RUN uv sync --frozen --no-dev

CMD ["llmclick", "serve", "--host", "0.0.0.0", "--port", "8000"]

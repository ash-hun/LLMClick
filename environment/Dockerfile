FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.13 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv PATH="/app/.venv/bin:$PATH"

# dependency layer: only the lock and project metadata
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py config.py ./
COPY core/ ./core/
COPY jeff/ ./jeff/
COPY configs/ ./configs/
RUN uv sync --frozen --no-dev

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

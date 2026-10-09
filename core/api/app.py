"""The FastAPI application: `llmclick serve`, or `uvicorn core.api.app:app --host 0.0.0.0 --port 8000`."""

import logging

from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from core.settings import ENV_FILE

load_dotenv(ENV_FILE)  # Hugging Face downloads read HF_TOKEN from the environment
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from core.api.guard import authorized  # noqa: E402
from core.api.routers import config_channel_router, job_channel_router, system_channel_router  # noqa: E402

app = FastAPI(title="LLMClick Pipeline", version="0.1.0",
              description="Config-driven custom model building: one YAML per model, one pipeline per recipe.")


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


for router in (config_channel_router, job_channel_router, system_channel_router):
    app.include_router(router, dependencies=[Depends(authorized)])  # /health stays open for the container healthcheck

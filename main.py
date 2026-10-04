"""LLMClick API entry point: `uvicorn main:app --host 0.0.0.0 --port 8000`."""

import logging

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv("environment/.env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from core.api.routers import config_channel_router, job_channel_router, system_channel_router  # noqa: E402

app = FastAPI(title="LLMClick Pipeline", version="0.1.0",
              description="Config-driven custom model building: one YAML per model, one pipeline per recipe.")


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(config_channel_router)
app.include_router(job_channel_router)
app.include_router(system_channel_router)

"""The FastAPI application: `llmclick serve`, or `uvicorn core.api.api_app:app --host 0.0.0.0 --port 8000`."""

from fastapi import Depends, FastAPI

from core.settings import prepare_process

prepare_process()  # before the routers import the channels: uvicorn starts here, not in the CLI

from core.api.access_guard import authorized  # noqa: E402
from core.api.routers import config_channel_router, evaluation_channel_router, job_channel_router, system_channel_router  # noqa: E402

app = FastAPI(title="LLMClick Pipeline", version="0.1.0",
              description="Config-driven custom model building: one YAML per model, one pipeline per recipe.")


@app.get("/health", tags=["Health"], summary="Liveness check",
         description="Open without a token, for the container healthcheck.")
def health() -> dict[str, str]:
    return {"status": "ok"}


for router in (config_channel_router, job_channel_router, evaluation_channel_router, system_channel_router):
    app.include_router(router, dependencies=[Depends(authorized)])  # /health stays open for the container healthcheck

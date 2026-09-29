"""System Channel: what this server can run (registries) and in which order (stages)."""

from fastapi import APIRouter

from core.config.schema import STAGES
from core.registry import catalogue

system_channel_router = APIRouter(prefix="/api/system", tags=["System Channel"])


@system_channel_router.get("/registries")
def registries() -> dict[str, list[str]]:
    return catalogue()


@system_channel_router.get("/stages")
def stages() -> list[str]:
    return list(STAGES)

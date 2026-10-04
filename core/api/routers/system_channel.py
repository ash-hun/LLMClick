"""System Channel: what this server can build (recipes, their stages and registry keys)."""

from fastapi import APIRouter

from core.registry import catalogue

system_channel_router = APIRouter(prefix="/api/system", tags=["System Channel"])


@system_channel_router.get("/recipes")
def recipes() -> dict[str, dict[str, list[str]]]:
    return catalogue()

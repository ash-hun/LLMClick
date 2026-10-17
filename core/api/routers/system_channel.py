"""System Channel: what this server can build (recipes, their stages and registry keys)."""

from fastapi import APIRouter

from core.registry import catalogue

system_channel_router = APIRouter(prefix="/api/system", tags=["System Channel"])


@system_channel_router.get("/recipes", summary="List the recipes",
                           description="Per recipe: its stages and the keys its registries accept (sources, teachers, ...).")
def recipes() -> dict[str, dict[str, list[str]]]:
    return catalogue()

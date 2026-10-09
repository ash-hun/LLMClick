"""What the API lets a caller do: a shared token when one is configured, and only the directories in `API_PATHS`."""

from pathlib import Path
from typing import Annotated

from fastapi import Header, HTTPException

from core.settings import get_settings


def authorized(authorization: Annotated[str | None, Header()] = None) -> None:
    """Dependency of every /api route: with `API_TOKEN` set, the request must carry `Authorization: Bearer <token>`;
    without it the API is open, as on a machine only its owner can reach."""
    token = get_settings().API_TOKEN
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Authorization: Bearer <API_TOKEN> is required",
                            headers={"WWW-Authenticate": "Bearer"})


def confined(path: str, what: str) -> None:
    """403 unless `path` resolves inside one of `API_PATHS`. Resolving follows `..` and symlinks, so a config cannot
    name a file or directory outside the roots however it spells the path."""
    roots = get_settings().api_roots
    real = Path(path).resolve()
    if not any(real == root or root in real.parents for root in roots):
        raise HTTPException(status_code=403, detail=f"{what} {path!r} is outside API_PATHS {[str(r) for r in roots]}")

"""`uvicorn main:app` still works from the repository root; the application lives in the package (core.api.app)."""

from core.api.app import app

__all__ = ["app"]

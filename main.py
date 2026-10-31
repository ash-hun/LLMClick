"""`uvicorn main:app` still works from the repository root; the application lives in the package (core.api.api_app)."""

from core.api.api_app import app

__all__ = ["app"]

"""Process-level settings read from environment/.env; per-experiment settings live in the YAML config instead."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="environment/.env", env_file_encoding="utf-8", extra="ignore")

    HF_TOKEN: str = ""
    JOBS_DB: str = "./output/_jobs.sqlite"   # the API's job table
    JOB_WORKERS: int = 1                     # jobs run at once; set it to the number of GPUs on a multi-GPU server
    API_TOKEN: str = ""                      # when set, every /api route needs `Authorization: Bearer <token>`
    API_PATHS: str = "."                     # directories (colon-separated) the API may read configs and data from and write output to

    @property
    def api_roots(self) -> list[Path]:
        return [Path(root).resolve() for root in self.API_PATHS.split(":") if root]


@lru_cache
def get_settings() -> Settings:
    return Settings()

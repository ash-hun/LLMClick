"""Process-level settings: the environment, plus the file `LLMCLICK_ENV` names (default `environment/.env`, read from
the working directory). Per-experiment settings live in the YAML config instead."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = os.environ.get("LLMCLICK_ENV", "environment/.env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

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

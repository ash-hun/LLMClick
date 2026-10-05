"""Process-level settings read from environment/.env; per-experiment settings live in the YAML config instead."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="environment/.env", env_file_encoding="utf-8", extra="ignore")

    HF_TOKEN: str = ""
    JOBS_DB: str = "./output/_jobs.sqlite"   # the API's job table
    JOB_WORKERS: int = 1                     # jobs run at once; set it to the number of GPUs on a multi-GPU server


@lru_cache
def get_settings() -> Settings:
    return Settings()

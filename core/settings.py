"""Process-level settings: the environment, plus the file `LLMCLICK_ENV` names (default `environment/.env`, read from
the working directory). Per-experiment settings live in the YAML config instead."""

import os
import sys
import logging
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = os.environ.get("LLMCLICK_ENV", "environment/.env")
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"  # the console and every pipeline.log


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    HF_TOKEN: str = ""
    JOBS_DB: str = "./output/_jobs.sqlite"   # the API's job table
    JOB_WORKERS: int = 1                     # jobs run at once; set it to the number of GPUs on a multi-GPU server
    API_TOKEN: str = ""                      # when set, every /api route needs `Authorization: Bearer <token>`
    API_PATHS: str = "."                     # directories (colon-separated) the API may read configs and data from and write output to
    PIPER_PYTHON: str = "environment/piper/.venv/bin/python"  # the Piper program's environment (environment/piper/setup.sh)
    QWEN_PYTHON: str = "environment/qwen/.venv/bin/python"    # Qwen3-TTS on MLX, Apple silicon (environment/qwen/setup.sh)

    @property
    def api_roots(self) -> list[Path]:
        return [Path(root).resolve() for root in self.API_PATHS.split(":") if root]


def prepare_process() -> None:
    """What every entry point (CLI, API server) does first: the env file into the environment, where Hugging Face
    downloads read HF_TOKEN, and log records to stderr. Safe to call again: neither step repeats itself."""
    load_dotenv(ENV_FILE)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)


@lru_cache
def get_settings() -> Settings:
    return Settings()

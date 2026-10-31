"""`framework` core/settings.py: every entry point prepares the process the same way, and doing it twice changes nothing."""

import os
import logging
from pathlib import Path

import pytest

from core import settings


def test_prepare_process_loads_the_env_file_and_logging_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("LLMCLICK_TEST_TOKEN=from-the-file\n")
    monkeypatch.setattr(settings, "ENV_FILE", str(env))
    monkeypatch.setenv("LLMCLICK_TEST_TOKEN", "")  # recorded, so the variable is removed again after the test
    monkeypatch.delenv("LLMCLICK_TEST_TOKEN")
    settings.prepare_process()
    handlers = list(logging.getLogger().handlers)
    settings.prepare_process()
    assert os.environ["LLMCLICK_TEST_TOKEN"] == "from-the-file" and logging.getLogger().handlers == handlers

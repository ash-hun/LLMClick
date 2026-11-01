"""`framework` core/logger.py: one root handler however often it is set up, and secrets never reach the rendered output."""

import io
import json
import logging
import warnings

import pytest

from core import logger


@pytest.fixture
def fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logger, "_configured", False)
    monkeypatch.setattr(logging.getLogger(), "handlers", list(logging.getLogger().handlers))
    monkeypatch.setattr(logging.getLogger(), "level", logging.getLogger().level)


def test_configure_logging_twice_keeps_one_root_handler(fresh: None) -> None:
    logger.configure_logging()
    logger.configure_logging()

    assert len(logging.getLogger().handlers) == 1


def test_sensitive_keys_are_masked_in_the_rendered_line(fresh: None) -> None:
    logger.configure_logging(level="INFO", fmt="json")
    handler, stream = logging.getLogger().handlers[0], io.StringIO()
    assert isinstance(handler, logging.StreamHandler)
    handler.setStream(stream)

    logger.get_logger("test").info("login_failed", password="hunter2", headers={"Authorization": "Bearer abc"})

    assert "hunter2" not in stream.getvalue() and "abc" not in stream.getvalue() and "login_failed" in stream.getvalue()


def test_python_warnings_render_as_json_records(fresh: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(logging, "_warnings_showwarning", None)
    logger.configure_logging(level="INFO", fmt="json")
    handler, stream = logging.getLogger().handlers[0], io.StringIO()
    assert isinstance(handler, logging.StreamHandler)
    handler.setStream(stream)

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn("triton unavailable", UserWarning, stacklevel=1)
    logging.captureWarnings(False)

    record = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert record["logger"] == "py.warnings" and "triton unavailable" in record["event"]

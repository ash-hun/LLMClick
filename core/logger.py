"""Logging for every entry point and module: structlog over the root logger, console or JSON by `LOG_FORMAT`."""

import os
import sys
import logging
from typing import Any
from pathlib import Path
from collections.abc import MutableMapping

import structlog
from structlog.dev import RichTracebackFormatter
from structlog.tracebacks import ExceptionDictTransformer

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "console"   # `json` in production: collectors read it, and a value's newline cannot forge a line
REDACTED = "***"
SENSITIVE_KEYS: frozenset[str] = frozenset({
    "api_key", "api_token", "authorization", "cookie", "hf_token", "password", "secret", "set-cookie", "token",
})

_configured: bool = False


def _mask(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: REDACTED if str(k).lower() in SENSITIVE_KEYS else _mask(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_mask(v) for v in value)
    return value


def _redact(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        event_dict[key] = REDACTED if key.lower() in SENSITIVE_KEYS else _mask(event_dict[key])
    return event_dict


SHARED: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    _redact,
]


def _formatter(fmt: str, colors: bool) -> logging.Formatter:
    # Tracebacks carry no local variables: they are the commonest way a secret reaches a log.
    renderer: list[Any] = (
        [structlog.processors.ExceptionRenderer(ExceptionDictTransformer(show_locals=False)),
         structlog.processors.JSONRenderer()]
        if fmt == "json"
        else [structlog.dev.ConsoleRenderer(colors=colors,
                                            exception_formatter=RichTracebackFormatter(show_locals=False))]
    )
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=SHARED,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, *renderer],
    )


def log_format() -> str:
    return (os.environ.get("LOG_FORMAT") or DEFAULT_LOG_FORMAT).lower()


def _route_to_stdlib() -> None:
    # Without this structlog prints straight to stdout; with it every record goes through the root logger's handlers.
    if not structlog.is_configured():
        structlog.configure(
            processors=[*SHARED, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """The root logger's one handler, on stderr; entry points call it, and calling it again adds no handler."""
    global _configured
    if _configured:
        return
    _route_to_stdlib()
    # stderr, not stdout: the CLI prints its JSON result on stdout and callers parse it.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_formatter(fmt or log_format(), colors=sys.stderr.isatty()))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel((level or os.environ.get("LOG_LEVEL") or DEFAULT_LOG_LEVEL).upper())
    _configured = True


def file_handler(path: Path) -> logging.Handler:
    """A handler that appends records to `path` in the process's log format, without terminal colours."""
    handler = logging.FileHandler(path)
    handler.setFormatter(_formatter(log_format(), colors=False))
    return handler


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Safe at import time: it leaves the root logger's handlers to `configure_logging`."""
    _route_to_stdlib()
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger

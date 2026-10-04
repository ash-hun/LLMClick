"""Content hashing and JSON helpers shared by stages and manifests."""

import fcntl
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on `path`; a second thread or process waits here until the first is done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield

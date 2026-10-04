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


SMALL_FILE = 1 << 20


def directory_signature(path: Path) -> list[Any] | None:
    """What a directory of model files holds, cheaply: where it really is (links resolved; stage directories are
    named by fingerprint, so that alone tells two builds apart), every file's size, and the content of small files
    such as configs. None when it is not a directory, e.g. a Hub model ID or a checkpoint not built yet."""
    if not path.is_dir():
        return None
    real = path.resolve()
    files = sorted(file for file in real.rglob("*") if file.is_file())
    return [str(real), [[str(file.relative_to(real)), file.stat().st_size,
                         sha256_file(file) if file.stat().st_size < SMALL_FILE else None] for file in files]]

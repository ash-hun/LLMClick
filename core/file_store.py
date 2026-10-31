"""File helpers shared by stages and manifests: content hashes, atomic writes, JSON, locks."""

import fcntl
import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """A temporary path beside `path` to write into, moved onto `path` only when the block finishes: a reader never
    sees half a file, and a write that fails leaves no temporary behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        yield temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(path)


@contextmanager
def staged_directory(path: Path) -> Iterator[Path]:
    """An empty directory beside `path` to fill, renamed to `path` only when the block finishes, so `path` exists
    whole or not at all. A staging directory a crash left behind is cleared first."""
    staging = path.with_name(path.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    yield staging
    staging.rename(path)


def write_json(path: Path, value: Any) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[Any]:
    """One JSON value per line; blank lines are skipped."""
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[Any]) -> None:
    """The whole file at once, through a temporary name: a reader never sees a half-written file."""
    with atomic_path(path) as temporary:
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


@contextmanager
def locked(path: Path, on_wait: Callable[[], None] | None = None) -> Iterator[None]:
    """Hold an exclusive lock on `path`; a second thread or process waits here until the first is done. When the
    lock is taken, `on_wait` is called once before waiting, so the wait can be shown instead of looking like a hang."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if on_wait is not None:
                on_wait()
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

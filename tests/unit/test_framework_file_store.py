"""`framework` core/file_store.py: a file or directory exists whole or not at all, and a rerun after a crash starts clean."""

from pathlib import Path

import pytest

from core.file_store import atomic_path, read_json, read_jsonl, staged_directory, write_json, write_jsonl


def test_atomic_path_replaces_only_a_finished_write_and_leaves_no_temporary(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "voice.wav"
    with atomic_path(target) as temporary:
        temporary.write_text("first")
        assert not target.exists()
    assert target.read_text() == "first" and not temporary.exists()
    with pytest.raises(RuntimeError):
        with atomic_path(target) as temporary:
            temporary.write_text("half")
            raise RuntimeError("died mid-write")
    assert target.read_text() == "first" and not temporary.exists()
    write_json(tmp_path / "a.json", {"b": 1})
    write_json(tmp_path / "a.json", {"b": 1})
    assert read_json(tmp_path / "a.json") == {"b": 1} and sorted(path.name for path in tmp_path.iterdir()) == ["a.json", "deep"]


def test_staged_directory_appears_whole_and_clears_what_a_crash_left(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint"
    with pytest.raises(RuntimeError):
        with staged_directory(target) as staging:
            (staging / "model.ckpt").write_text("half")
            raise RuntimeError("export died")
    assert not target.exists() and (tmp_path / "checkpoint.tmp" / "model.ckpt").exists()
    with staged_directory(target) as staging:
        assert list(staging.iterdir()) == []
        (staging / "voice.onnx").write_text("done")
    assert [path.name for path in target.iterdir()] == ["voice.onnx"] and not (tmp_path / "checkpoint.tmp").exists()


def test_jsonl_round_trip_skips_blank_lines_and_writes_whole_files(tmp_path: Path) -> None:
    rows = [{"text": "Jarvis", "n": 1}, {"text": "한국어", "n": 2}]
    write_jsonl(tmp_path / "out" / "rows.jsonl", rows)
    assert read_jsonl(tmp_path / "out" / "rows.jsonl") == rows
    (tmp_path / "gaps.jsonl").write_text('{"a": 1}\n\n  \n{"a": 2}\n')
    assert read_jsonl(tmp_path / "gaps.jsonl") == [{"a": 1}, {"a": 2}]
    assert [path.name for path in (tmp_path / "out").iterdir()] == ["rows.jsonl"]
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "missing.jsonl")

"""`tts_piper` modeling/tts/audio.py: the recordings a source names are found on disk, and missing ones are reported."""

from pathlib import Path

import pytest

from tests.recipe_inputs import TEXTS, tts_recordings as recordings
from modeling.config import Keyed, local_paths
from modeling.tuning.sources import SOURCES
from modeling.tts import audio


def test_recordings_of_a_source_are_found_on_disk(tmp_path: Path) -> None:
    root = recordings(tmp_path)
    source = Keyed(name="delimited", path=str(root / "metadata.csv"), audio_dir=str(root / "wavs"))
    found = list(audio.recordings(source))
    assert [number for number, _, _ in found] == list(range(len(TEXTS))) and all(path and path.is_file() for _, _, path in found)
    (root / "metadata.csv").write_text("utt0|ok\nmissing|no file\n")
    assert [path is None for _, _, path in audio.recordings(source)] == [False, True]
    assert audio.signatures([source, Keyed(name="huggingface")])[1] is None
    assert local_paths([source], audio.AUDIO_KEYS) == [str(root / "metadata.csv"), str(root / "wavs")]
    with pytest.raises(ValueError, match="Unknown source 'nowhere'; registered: \\['delimited'"):
        SOURCES.check("delimited", "nowhere")

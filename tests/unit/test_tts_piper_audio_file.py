"""`tts_piper` modeling/tts/audio_file.py: the recordings a source names are found on disk, and missing ones are reported."""

from pathlib import Path

import pytest

from tests.recipe_inputs import TEXTS, tts_recordings as recordings
from modeling.modeling_config import Keyed, local_paths
from modeling.tuning.row_sources import SOURCES
from modeling.tts import audio_file


def test_recordings_of_a_source_are_found_on_disk(tmp_path: Path) -> None:
    root = recordings(tmp_path)
    source = Keyed(name="delimited", path=str(root / "metadata.csv"), audio_dir=str(root / "wavs"))
    found = list(audio_file.recordings(source))
    assert [number for number, _, _ in found] == list(range(len(TEXTS))) and all(path and path.is_file() for _, _, path in found)
    (root / "metadata.csv").write_text("utt0|ok\nmissing|no file\n")
    assert [path is None for _, _, path in audio_file.recordings(source)] == [False, True]
    assert audio_file.signatures([source, Keyed(name="huggingface")])[1] is None
    assert local_paths([source], audio_file.AUDIO_KEYS) == [str(root / "metadata.csv"), str(root / "wavs")]
    with pytest.raises(ValueError, match="Unknown source 'nowhere'; registered: \\['delimited'"):
        SOURCES.check("delimited", "nowhere")

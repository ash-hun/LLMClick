"""`tts_piper` modeling/tts/quality_rater.py: DNSMOS cuts windows as Microsoft's script does and calibrates its outputs."""

import pytest
import numpy as np

from modeling.tts import quality_rater


def test_dnsmos_cuts_windows_as_microsofts_script_and_calibrates_its_outputs() -> None:
    second = np.arange(quality_rater.RATE, dtype=np.float32)
    windows = quality_rater.dnsmos_windows(second)  # doubled to 16 s, then 9.01 s windows one second apart
    assert windows.shape == (7, quality_rater.DNSMOS_WINDOW) and windows[1][0] == 0 and windows[1][-1] == second[quality_rater.DNSMOS_WINDOW % quality_rater.RATE - 1]
    assert len(quality_rater.dnsmos_windows(np.zeros(30 * quality_rater.RATE, dtype=np.float32))) == 7  # later windows end a sample short
    raw = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    assert quality_rater.calibrated(raw)["dnsmos_bak"] == pytest.approx(np.mean([-0.13166888 * x * x + 1.60915514 * x - 0.39604546 for x in (3, 4)]))

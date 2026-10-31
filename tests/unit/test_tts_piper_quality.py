"""`tts_piper` modeling/tts/quality.py: DNSMOS cuts windows as Microsoft's script does and calibrates its outputs."""

import pytest
import numpy as np

from modeling.tts import quality


def test_dnsmos_cuts_windows_as_microsofts_script_and_calibrates_its_outputs() -> None:
    second = np.arange(quality.RATE, dtype=np.float32)
    windows = quality.dnsmos_windows(second)  # doubled to 16 s, then 9.01 s windows one second apart
    assert windows.shape == (7, quality.DNSMOS_WINDOW) and windows[1][0] == 0 and windows[1][-1] == second[quality.DNSMOS_WINDOW % quality.RATE - 1]
    assert len(quality.dnsmos_windows(np.zeros(30 * quality.RATE, dtype=np.float32))) == 7  # later windows end a sample short
    raw = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    assert quality.calibrated(raw)["dnsmos_bak"] == pytest.approx(np.mean([-0.13166888 * x * x + 1.60915514 * x - 0.39604546 for x in (3, 4)]))

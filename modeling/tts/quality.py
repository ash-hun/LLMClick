"""How clean speech sounds without a reference: UTMOS naturalness and DNSMOS P.835 signal, background and overall."""

import urllib.request
from pathlib import Path

import numpy as np

from core.utils.files import atomic_path, sha256_file
from core.settings import get_settings
from modeling.tts import audio

RATE = 16000
# tarepan/SpeechMOS v1.2.0's utmos22_strong (MIT) exported to ONNX: the original needs torchaudio, which has no build
# for this torch; one 16 kHz clip per call, since the graph has no padding mask
UTMOS, UTMOS_REVISION, UTMOS_FILE = "TigreGotico/utmos-onnx", "ff41b8f440cb12ecda18261f9ff7326d058275ce", "utmos22_strong.onnx"
DNSMOS = "sig_bak_ovr.onnx"
DNSMOS_URL = ("https://raw.githubusercontent.com/microsoft/DNS-Challenge/82f1b17e7776a43eee395d0f45bae8abb700ad00/"
              "DNSMOS/DNSMOS/" + DNSMOS)
DNSMOS_SHA256 = "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"
DNSMOS_SECONDS = 9.01  # what the P.835 model reads; a shorter clip is repeated up to it
DNSMOS_WINDOW = int(DNSMOS_SECONDS * RATE)
# Microsoft's mapping of the raw P.835 outputs to MOS (DNSMOS/dnsmos_local.py), highest power first
CALIBRATION = {"dnsmos_sig": (-0.08397278, 1.22083953, 0.0052439),
               "dnsmos_bak": (-0.13166888, 1.60915514, -0.39604546),
               "dnsmos_ovrl": (-0.06766283, 1.11546468, 0.04602535)}
METRICS = ("utmos", *CALIBRATION)


def dnsmos_windows(samples: np.ndarray) -> np.ndarray:
    """The 9.01 s windows DNSMOS scores, one second apart, as dnsmos_local.py cuts them: a short clip is doubled
    until it fills a window."""
    while len(samples) < DNSMOS_WINDOW:
        samples = np.concatenate([samples, samples])
    hops = int(np.floor(len(samples) / RATE) - DNSMOS_SECONDS) + 1
    # the float window end leaves every window from the eighth on a sample short, and Microsoft's script skips them,
    # so only a clip's first 16 s count; kept so the scores match its own
    windows = [samples[hop * RATE: int((hop + DNSMOS_SECONDS) * RATE)] for hop in range(hops)]
    return np.stack([window for window in windows if len(window) == DNSMOS_WINDOW]).astype(np.float32)


def calibrated(raw: np.ndarray) -> dict[str, float]:
    """Raw P.835 outputs (windows x [signal, background, overall]) as the clip's mean MOS of each."""
    return {name: float(np.polyval(CALIBRATION[name], raw[:, column]).mean()) for column, name in enumerate(CALIBRATION)}


def ensure(directory: Path) -> Path:
    """The DNSMOS model under `directory`, downloaded once and checked against its pinned hash."""
    path = directory / DNSMOS
    if not (path.is_file() and sha256_file(path) == DNSMOS_SHA256):
        directory.mkdir(parents=True, exist_ok=True)
        with atomic_path(path) as temporary:
            urllib.request.urlretrieve(DNSMOS_URL, temporary)
            if sha256_file(temporary) != DNSMOS_SHA256:
                raise ValueError(f"{DNSMOS_URL} does not match its pinned sha256 {DNSMOS_SHA256}")
    return path


class Rater:
    """Both predictors on onnxruntime, loaded once for a stage's run."""

    def __init__(self, directory: Path) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        utmos = hf_hub_download(UTMOS, UTMOS_FILE, revision=UTMOS_REVISION, token=get_settings().HF_TOKEN or None)
        options = {"providers": ["CPUExecutionProvider"]}
        self.utmos = ort.InferenceSession(utmos, **options)
        self.dnsmos = ort.InferenceSession(str(ensure(directory)), **options)

    def __call__(self, path: Path) -> dict[str, float]:
        samples, rate = audio.read(path)
        clip = audio.resample(samples, rate, RATE)
        utmos = self.utmos.run(None, {"wave": clip[np.newaxis]})[0]
        raw = self.dnsmos.run(None, {"input_1": dnsmos_windows(clip)})[0]
        return {"utmos": float(utmos[0]), **calibrated(raw)}

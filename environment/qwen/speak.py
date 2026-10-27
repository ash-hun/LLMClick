"""Speak a batch of texts in a cloned voice with Qwen3-TTS: a JSON request on stdin, one wav per job."""

import os
import sys
import json

import numpy as np
import soundfile
import mlx.core as mx
from mlx_audio.tts.utils import load_model

# {"model", "revision", "reference", "reference_text", "jobs": [{"text", "wav", "seed"}]}; a job whose wav exists is
# skipped, so a rerun after a crash speaks only what is missing, and a seeded job gives the same file every time
request = json.load(sys.stdin)
jobs = [job for job in request["jobs"] if not os.path.isfile(job["wav"])]
if jobs:
    model = load_model(request["model"], revision=request["revision"])
    for job in jobs:
        mx.random.seed(job["seed"])
        results = model.generate(text=job["text"], ref_audio=request["reference"], ref_text=request["reference_text"])
        samples = np.concatenate([np.array(result.audio) for result in results])
        os.makedirs(os.path.dirname(job["wav"]), exist_ok=True)
        temporary = job["wav"] + ".partial.wav"
        soundfile.write(temporary, samples, model.sample_rate, subtype="PCM_16")
        os.replace(temporary, job["wav"])
        print(f"{len(samples) / model.sample_rate:.2f}s {job['wav']}", flush=True)

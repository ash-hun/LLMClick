---
title: Speech datasets
description: data_speech, from seeds to a Piper training set, and why each stage is safe to rerun.
---

# Speech datasets: `data_speech`

Five stages take seeds to a directory Piper can train on. The first two are those of `data_synthetic`: seeds, and the
texts made from them. Then a Piper voice reads each text aloud, the recordings are measured and transcribed, and the
ones whose transcript matches their text are written out with a manifest that carries the voice's terms.

Every recording and every transcript is kept under its natural key, so a run that stops halfway continues where it
stopped, and a run with stricter checks reuses the recordings it already has.

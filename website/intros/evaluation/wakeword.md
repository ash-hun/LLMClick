---
title: Wake word detectors
description: evaluation_wakeword measures a detector's miss rate under noise and its false accepts per hour.
---

# Wake word detectors: `evaluation_wakeword`

A wake word detector fails two ways: it misses the name, or it wakes on something else. This recipe measures both
on what the detector never trained on, as wake word detectors are compared elsewhere: the miss rate under noise and
the false accepts per hour of audio that never says the name, at every threshold.

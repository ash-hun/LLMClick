---
title: Wake word detectors
description: evaluation_wakeword measures a detector's miss rate under noise and its false accepts per hour.
---

# Wake word detectors: `evaluation_wakeword`

A wake word detector fails two ways: it misses the name, or it wakes on something else. This recipe measures both on
what the detector never trained on, the way wake word detectors are compared elsewhere.

| Number | Measured on | Target |
|---|---|---|
| miss rate (FRR), per noise condition | the name said by voices the detector never heard, clean and at 10 and 5 dB | under 5% (openWakeWord) |
| false accepts per hour | hours of speech that never says the name, as one long stream | 0.5 an hour (openWakeWord), 0.1 (Picovoice) |
| negatives fired | near misses and commands said by unseen voices | as low as the sweep allows |
| the sweep | all three at every threshold from 0.01 to 0.99 | the miss rate at each target |

`wakeword`'s own `validate` says whether a detector passed on a held-out split of its training voice; this recipe
says how it will do in the room.

## Config and details

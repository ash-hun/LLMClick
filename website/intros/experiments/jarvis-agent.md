---
title: Jarvis voice agent
description: A wake word detector for "Jarvis" and a fine-tuned Jarvis voice with nine read-backs, built and checked with wakeword, tts_piper and data_speech; every iteration and failure.
---

# Experiment: a Jarvis voice agent

Two models for a voice agent, built end to end with the recipes of this repository on one Apple M4 Pro (48 GB,
no CUDA): the keyword spotter that wakes the agent when it hears "Jarvis", and the voice that answers. This page
records the results, and also each iteration that failed and why, because those runs are where most of the lessons
came from.

```
mic -> [KWS: always on, tiny] -> "Jarvis" -> [VAD + ASR: the command] -> LLM / agent -> [TTS: the reply]
          wakeword                                (not built here)        (not built here)    tts_piper
```

All listening checks use Whisper `openai/whisper-base` (revision `e37978b9`, English).

## Results

| Model | Experiment | Validation |
|---|---|---|
| KWS: "Jarvis" (`jarvis.onnx`, 204 KB, plus openWakeWord's 2.4 MB front end) | jarvis-wake-b6 | passed at threshold 0.5: recall 1.000, false_accepts 0.047, scenario 0.963 |
| TTS: Jarvis voice (`voice.onnx`, 64 MB) with nine read-backs | jarvis-medium-ft (best of 5 takes) | passed: wer 0.072 (source recordings 0.072), phrase_wer 0.000 |

### Wake word: every iteration

| Iteration | Phrase | Training data (positive / negative recordings) | Change | Threshold | recall | false_accepts | scenario | Gate |
|---|---|---|---|---|---|---|---|---|
| A2-1 (trial) | "Jarvis, wake up" | 25 / 312 | first run | 0.5 | 0.000 | 0.000 | 0.000 | failed (test bug) |
| A2-2 (trial) | "Jarvis, wake up" | 25 / 312 | test streams start with 2 s of noise | 0.5 | 0.833 | 0.000 | 1.000 | failed (recall) |
| A2-3 (trial) | "Jarvis, wake up" | 25 / 312 | silence after the phrase is a negative too | 0.5 | 0.833 | 0.013 | 1.000 | failed (recall) |
| B3 | "Jarvis" | 426 / 910 | full data | 0.5 | 1.000 | 0.057 | 0.925 | failed (false_accepts) |
| B3 | "Jarvis" | 426 / 910 | same model | 0.95 | 1.000 | 0.039 | 0.953 | passed |
| B6 | "Jarvis" | 426 / 2830 | + 2400 near misses by 120 new speakers | 0.5 | 1.000 | 0.047 | 0.963 | passed |
| B6, retrained | "Jarvis" | 426 / 2830 | windows keyed by recording content, not path | 0.5 | 1.000 | 0.088 | 0.953 | failed (false_accepts) |
| **B7** | "Jarvis" | 426 / 2830 | negatives weighted 100, cosine learning rate | 0.5 | **1.000** | **0.037** | **0.981** | **passed** |

Each row is measured on its own held-out set, so rows of different data do not compare directly. The fair
comparison of B3 and B6 uses only the recordings neither trained on:

| Model | Threshold | "Jarvis" detected | Near misses fired | Commands and ordinary speech fired |
|---|---|---|---|---|
| B3 | 0.50 | 107/107 | 123/582 (21.1%) | 0/126 |
| B3 | 0.95 | 107/107 | 92/582 (15.8%) | 0/126 |
| B6 | 0.50 | 107/107 | 33/582 (5.7%) | 0/126 |
| B6 | 0.95 | 107/107 | 21/582 (3.6%) | 0/126 |

### Voice: every iteration

| Iteration | Voice | Read-backs taken | phrase_wer | Read-backs exact | Gate |
|---|---|---|---|---|---|
| A4 (trial) | jgkawell/jarvis medium, unmodified | 1 take | 1.25 ("At your service, sir.") | 1 of 2 tried | (no gate) |
| B5 | fine-tuned, 100 epochs | 1 take | 0.667 | 6 of 9 | failed |
| **B5b** | same voice | best of 5, "Target. Analysis." | **0.000** | **9 of 9** | **passed** |

| Read-back | Text | B5: one take heard as | B5b: exact takes of 5 |
|---|---|---|---|
| wake_up | At your service, sir. | at your service, sir. | 3 |
| web_search | Searching Reference. | Searching Reference. | 5 |
| horizontal_task | Processing Long Task. | Processing Luntask. | 3 |
| remember_pattern | Checking the Pattern. | Checking the pattern. | 5 |
| analysis | Target Analysis. / Target. Analysis. | Tubby analysis. | 2 (as "Target. Analysis.") |
| focus_mode_enter | Activate Focus mode. | activate focus mode. | 5 |
| focus_mode_quit | Deactivate Focus mode. | deactivate focus mode. | 5 |
| persona_karen | Change to KAREN. | Change to Karen. | 3 |
| persona_jarvis | Change to JARVIS. | Change to jobless. | 1 |

## The data

Synthetic speech throughout, made by `data_speech` with Piper voices and kept or dropped by Whisper.

| Set | Voice | Content | Readings | Kept | Time |
|---|---|---|---|---|---|
| wake-phrase | libritts_r (multi-speaker), speakers 0-299 | "Jarvis." / "Jarvis?" / "Jarvis!" | 900 | 533 (59%), 260 speakers | 12.5 min |
| wake-negatives | libritts_r, speakers 0-39 | 14 near misses, 10 commands said alone, 6 ordinary lines | 1200 | 1138 (95%) | 16.7 min |
| wake-nearmiss | libritts_r, speakers 40-159 | 20 near misses, no listening check | 2400 | 2400 | 28.6 min |
| jarvis-lines | jgkawell/jarvis medium | 40 lines from a local qwen3.5:0.8b (8 situations) + 18 fixed lines | 58 | 48 (250 s) | about 1 min |

What Whisper heard in the rejected "Jarvis" readings: "Dervis." 142, "Dervis?" 99, "Service." 45, "Dervus." 32.
One word gives the recognizer no context, so part of these readings may be fine; the kept set is the conservative
one.

## Step by step

### Trial runs (wake phrase "Jarvis, wake up")

A small run on real models before the full scale: 80 speakers read the phrase (31 kept, 39%: "Dervus", "Service",
and Whisper writing "Wakeup" as one word), 16 speakers read 26 other lines (390 kept).

- **Recall 0 was a test bug.** The detector scored its training positives at 0.99996 and negatives at 3e-5, and a
  clean 2-second window of a held-out phrase at 1.0, yet it never fired in validation. The test stream was 0.5 s of
  noise, the phrase and 0.6 s: about 2 s in all, while the detector reads the last 2 s, so the stream held one window
  or none. An always-on microphone always has the past 2 s; the streams now start with 2 s of noise.
- **A second firing after the phrase.** With tones in the unit test, the detector fired again 0.6 s after the
  phrase, in the silence before the command: "phrase, then silence" had never been a training example, only
  "phrase, then speech". The `followed` negatives now end anywhere from 0.5 s after the phrase.
- **The model works in openWakeWord's own runtime.** The exported file loaded unchanged in `openwakeword==0.6.0`
  and fired at the same moment as this recipe's scorer (2.80-3.20 s against 2.87 s, phrase ending at 2.91 s), and
  not during the command that followed.
- **The reference voice garbles one read-back.** jgkawell/jarvis read "At your service, sir." as "I trust you all
  so, sir." (WER 1.25), and "Searching Reference." exactly.

Then the wake word became the name alone: "Jarvis" wakes the agent, and "wake up" is one of the commands that
follow it.

### B3: the first full detector fails on near misses

426 positives and 910 negatives for training, 107 and 228 held out; 1704 positive and 4692 negative windows;
3000 steps, 68 s for the whole recipe. At threshold 0.5, false_accepts 0.057, over the maximum 0.05.

Every false accept was a near miss of the name: "Jars." 3 of 3, "Service." 3 of 7, "Charges." 3 of 12,
"Nervous." 2 of 10, "Jasmine." 1 of 4, "Thank you, Travis." 1 of 9. Commands and ordinary lines: none. The 8
failed scenarios were the same thing, a near miss after "Jarvis" that fired a second time.

| Threshold | 0.50 | 0.70 | 0.80 | 0.90 | 0.95 | 0.98 | 0.99 |
|---|---|---|---|---|---|---|---|
| recall | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| false_accepts | 0.057 | 0.053 | 0.053 | 0.053 | 0.039 | 0.039 | 0.035 |
| scenario | 0.925 | 0.925 | 0.935 | 0.953 | 0.953 | 0.953 | 0.963 |

The scores are saturated near 0 and 1, so the threshold barely moves the errors. It passes at 0.95, but the
problem is the data: for this synthetic voice "Jarvis" and "Service" are close enough that Whisper confused them.

### B4 and B5: the voice can only learn what it already says

The voice's training set is read by the voice itself. 10 of 58 lines failed the listening check, among them the name
("Javis Hietzer", "Ciao for CIS", "Change to Javas", "job with Susback") and two read-backs ("Processing Lump task",
"Talky to Nalysis"). The lines the voice says badly are exactly the ones the check removes, so fine-tuning on this
set cannot teach them.

Fine-tuning ran on the M4 Pro's GPU through MPS: 100 epochs on 46 clips in about 6.5 min. wer 0.070 against the
recordings' 0.072, duration ratio 0.999 (2 held-out lines: enough to show the voice did not break, not more).
phrase_wer 0.667: 6 of 9 read-backs right. "At your service, sir." was now right.

### B5b: one take is a coin toss

Piper draws its noise unseeded, so the same text gives a different recording every time, and the validate stage
kept the first. Five takes of each read-back: 8 of 9 have at least one exact take; "Target Analysis." has none.
Rewritten, "Target. Analysis." (a pause between the words) comes out right, while "Target analysis.", "Targette
Analysis." and "Run target analysis." do not. `validation.phrase_takes` now keeps the best-heard take; with five
takes and the rewritten text every read-back is exact. Only validation changed, so the trained voice was reused and
the stage took 3.5 min.

### B6: near misses fix the detector

20 near misses ("Travis", "Davis", "Harvest", "Service", "Darvish", "Charges", "Marvis", "Jars", "Garbage",
"Karen", "Jasmine", "Nervous", "Jarrett", "Harvey", "Carvings", "Starbucks", "Purpose", "Office", "Dervish",
"Jockeys") by 120 speakers not in the other negatives, without a listening check: a reading Whisper mishears is still
not "Jarvis". Added as a second negative source, nothing else changed: 12372 negative windows, 2 min for the recipe,
final loss 3.9e-4 against B3's 6.0e-5 (harder negatives, less saturation). Passed at threshold 0.5, and on the
recordings neither model trained on, near-miss false accepts fell from 21.1% to 5.7% with no loss in detection.
What remains are words that begin like the name: "Jars", "Jarrett", "Charges".

In openWakeWord's runtime, on "Jarvis" followed by "Wake up.", by "Please search to the paper reference on the
web." and by "Enter the focus mode.", B6 fires as the name ends (scores 0.94-1.00 from 2.56 to 3.04 s, the name
ending at about 2.65 s) and not during the command.

### B7: a recipe that passes on any seed, not on a lucky one

A fix made each training window's noise depend on the recording's content rather than its path (until then, moving
the recordings changed the windows). With new windows and nothing else changed, B6's recipe failed: false_accepts
0.088. Scoring the B6 detector itself with the new test noise gave 0.049, so the test had barely moved; the detector
had. Over seeds (each seed also draws its own validation split, so the gate's 5% is measured six times on different
held-out sets):

| Training | false_accepts over seeds | Worst | Mean | recall |
|---|---|---|---|---|
| B6 recipe (constant rate, equal weights) | 0.037, 0.051, 0.088 | 0.088 | 0.059 | 1.000 |
| 8 rounds of augmentation | 0.042, 0.068, 0.151 | 0.151 | 0.087 | 1.000 |
| layer_dim 64 | 0.061, 0.064, 0.071 | 0.071 | 0.065 | 1.000 |
| cosine rate | 0.042, 0.051, 0.062 | 0.062 | 0.052 | 1.000 |
| cosine, negatives weighted 30 | 0.023 to 0.058 (6 seeds) | 0.058 | 0.040 | 1.000 |
| **cosine, negatives weighted 100 (B7)** | 0.031 to 0.047 (6 seeds) | **0.047** | **0.037** | **1.000** |
| cosine, negatives weighted 300 | 0.028 to 0.045 (6 seeds) | 0.045 | 0.035 | 1.000 |

More augmentation widened the spread rather than narrowing it, so data was not what was missing. Two things were: a
constant learning rate leaves the detector wherever step 3000 happens to land, and with recall at 1.000 on every seed
the loss had room to care more about negatives. 100 rather than 300 because the two differ by less than the test can
tell (about 0.008 standard error at 708 negatives) and recall on real voices is not measured. Both are settings of
`training` (`negative_weight`, `schedule`); their defaults keep the old behaviour, because on a small set weight 100
leaves too few positives in each batch's loss to learn the phrase at all.

### Evaluation: voices neither detector heard

`evaluation_wakeword` (configs/evaluation/wakeword.yaml) on another Piper voice, VCTK: 161 readings of the name that
Whisper heard as "Jarvis" (of 327, by 109 speakers), 2790 near misses, commands and ordinary sentences, and 5.4 hours
of LibriSpeech test-clean for false accepts per hour. Gaussian noise, threshold 0.5, refractory 1 s:

| Detector | Missed, clean | Missed, 10 dB | Missed, 5 dB | Negatives fired, clean | False accepts per hour |
|---|---|---|---|---|---|
| B6 | 7/161 (4.3%) | 66/161 (41.0%) | 76/161 (47.2%) | 16.1% | 3.1 (17 in 5.4 h) |
| B7 | 59/161 (36.6%) | 44/161 (27.3%) | 66/161 (41.0%) | 1.0% | 9.6 (52 in 5.4 h) |

Neither comes near openWakeWord's target (under 5% missed at 0.5 false accepts per hour): at no threshold does
either reach 0.5 per hour. The two validation splits said 100% recall; they were readings by the training voice.
B7's weighting of negatives made it refuse near misses ("Jarrett" fired 96 of 103 times on B6, 13 on B7) and, with
them, much of a voice it never heard: on the missed readings its highest score is 0.011 at the median, so it is not
a threshold away. Both lose most of the name under noise. What this points to: positives from more voices than one
Piper model (other Piper voices, real recordings), real noise and room reverberation in training, and ambient
speech as negatives.

## What the runs taught

1. A test stream must be at least as long as the detector's window, or a test bug looks like a model that hears
   nothing.
2. "Name, then something else", silence included, must be a negative example, or the detector fires twice in the
   agent's first moment.
3. A wake word's remaining errors are near misses of the name. Near-miss data fixes them; the threshold barely does.
4. For negatives, the listening check throws away good data: anything that is not the name is a negative.
5. A voice fine-tuned on its own output cannot learn the lines it says badly; recordings that say them right are
   what would.
6. A Piper take is a draw. Keep the best-heard of several takes of anything the agent plays as a file.
7. One passing run is not a passing recipe. Run the gate over a few seeds before trusting a margin of a point.
8. A held-out split of the training voice measures memory of that voice. Score on voices it never heard before
   calling a detector done.

## Limits

- The positives are 260 speakers of one synthetic voice, split by recording rather than by speaker, so a held-out
  speaker's other readings are probably in training and recall is probably optimistic.
- Noise is Gaussian at 20 dB SNR; there is no room reverberation and no real background audio.
- false_accepts is a share of chosen hard negatives, not a rate per hour of everyday audio.
- The voice's held-out set is 2 lines; its `wer` says only that the voice did not break.
- VAD, ASR and command recognition are not part of this experiment.

## Reproducing

Configs: `data/wake_phrase.yaml`, `data/wake_negatives.yaml`, `data/wake_nearmiss.yaml`, `wakeword/jarvis.yaml`,
`data/speech.yaml`, `tts/piper_jarvis.yaml`, run in that order with each `<hash>` replaced by the data experiment's
key (`llmclick validate <config>`). Piper's noise is unseeded, so recordings differ between runs: the counts above
will move a little, the conclusions should not.

---
title: Wake word
description: wakeword trains the detector that fires when the agent hears its phrase, and tests it on phrase-then-command streams.
---

# Wake word: `wakeword`

A voice agent listens all the time, but it should act only after hearing its name, such as "Jarvis". This recipe
trains the small detector that listens for that name; the command that follows it ("wake up", "search the web") is
heard by speech recognition afterwards. It uses openWakeWord's frozen speech features and
trains only a small scoring head on top of them, so a few hundred synthetic recordings are enough. The exported model
loads in openWakeWord's runtime.

Validation plays the agent's first moment back to the detector. The user says the name and then something else, with
noise underneath, and the detector must fire exactly once, on the name and not on what follows. The last section
shows how this recipe, the voice from `tts_piper`, and the agent itself fit together.

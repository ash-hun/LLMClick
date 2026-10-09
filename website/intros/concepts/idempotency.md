---
title: Why runs are safe to repeat
description: The design rule behind caching, resume and reruns.
---

# Why runs are safe to repeat

LLMClick is built around one rule: running the same thing twice must leave the same state as running it once. The
rule is there for recovery. A long run that dies at step three of five is normal, and if a rerun could duplicate
work, double-charge a teacher API or mix stale and fresh files, you would have to inspect the directory by hand
before daring to run again. With the rule, recovery is always the same command.

How it shows up:

- **Stage directories are named by content.** A stage's fingerprint covers its config sections, the content of local
  files it reads, the code version of the parts it uses and the fingerprints of the stages it reads. The directory
  exists with a `stage.json` only once the stage finished, so a half-built directory is rebuilt, never trusted.
- **Expensive work is cached per item.** Training keeps a resume snapshot and continues from it; the synthetic data
  recipe keeps every teacher call under a key made from the request, so a rerun after a crash makes only the
  missing calls and the budget check reads the cache as its ledger.
- **Files are written through a temporary name and renamed**, so a reader never sees a half-written file.
- **Randomness is seeded by position**, not by the clock: the batch order, a sampled completion, an evolution
  operator's draw all come from the config's seed and the item's key, so a resumed run repeats the uninterrupted one.
- **Locks make "is it built?" and "build it" one step**, so a CLI run and an API job asking for the same stage never
  build it twice; the second one says it is waiting and then reuses the result.

What this means for you: when something fails, read the error, fix the input, and run the same command. Do not
delete `output/` to "start clean"; the fingerprints already know what changed.

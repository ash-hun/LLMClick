---
title: Synthetic rows
description: data_synthetic, stage by stage, and why each one is safe to rerun.
---

# Synthetic rows: `data_synthetic`

Six stages take seeds to training rows. Read them as a funnel: the early stages make many candidates cheaply, the
later ones throw most of them away for a reason that is recorded. The published recipes the design follows (Magpie,
Tulu 3, OpenThoughts, Evol-Instruct, Nemotron) all agree on that shape: generate more than you keep, and let
verifiers and judges choose.

![The six stages](/img/data-synthetic-pipeline.png)

Before a large run, do a small one (two seeds, two answers) with the teacher you intend to use and read the cache
directories: they hold every request and answer, and they show quickly whether the teacher follows the generator's
format. A small teacher often does not; the filters will then reject most of its output, which is the pipeline
working as designed.

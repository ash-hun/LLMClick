---
title: How a run works
description: Fingerprints, caching, resume, locks, validation and tracking, in the order a run meets them.
---

# How a run works

Every recipe runs on the same runner, and the runner has a handful of rules that explain most of what you will
observe: why a second run is instant, why two configs share a data stage, why a crashed training continues instead
of restarting, and why a trained model is never handed on without passing validation.

The short version: a stage's directory is named by a hash of everything that decides its result, so "has this been
built?" is a directory lookup. The long version follows.

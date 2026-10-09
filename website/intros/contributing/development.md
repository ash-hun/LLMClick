---
title: Development
description: Tests, linting, typing, and the rules a change should follow.
---

# Development

The test suite runs every recipe end to end on a tiny random model, so it needs no download and no GPU and finishes in
about a minute. CI runs the linter, the type checker in strict mode and the tests on every pull request; the same
three commands run locally.

Rules a change should follow:

- A stage must be safe to run again in a directory that holds the files of an interrupted run: reuse or replace them,
  never fail on them.
- Anything that decides a stage's result goes into its fingerprint; anything that only says how to run (the
  tracker, the budget, the output directory) stays out.
- Bump a `version` when the code of a method, backbone, head or converter changes what the same inputs produce; the
  experiments that use it rebuild, every other cache stays.
- Documentation is generated from the README files and the code (this site included). Edit those, not the generated
  pages.

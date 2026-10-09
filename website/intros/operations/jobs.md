---
title: Running as a server
description: The HTTP API, background jobs, workers per GPU, cancellation and restarts.
---

# Running as a server

The CLI is enough for one person at one machine. The server exists for the other cases: a team submitting configs to
a shared GPU box, a UI polling progress, several jobs at once on several GPUs. `llmclick serve` starts a FastAPI
application with Swagger at `/docs`; `POST /api/jobs` with a config path or an inline config starts a run in the
background and returns a job you can poll, list and cancel.

What to know before relying on it:

- **Jobs survive restarts.** They live in a SQLite table. A job the server was running when it stopped is marked
  `interrupted`; submit the same config again and it continues from the stages already built.
- **One worker, one accelerator.** `JOB_WORKERS` sets how many jobs run at once; each worker owns one GPU. Keep it at
  1 on a single GPU or a Mac, set it to the number of GPUs otherwise.
- **The same config is the same job.** A job's id is the experiment key plus the stage plan, so resubmitting a
  finished job reruns it (from cache when nothing changed) and resubmitting a running one returns that job.
- **Cancel is cooperative.** `DELETE /api/jobs/{id}` asks the job to stop; it stops at its next progress report,
  keeps what it built, and continues from there when submitted again.
- **Every evaluation report is listable.** `GET /api/evaluations` returns the reports under an output directory with
  their headline numbers.

The endpoints, their parameters and what a repeated call does are in the API reference.

---
title: Securing the server
description: A shared token and a path allowlist, because the API reads files and starts training on your behalf.
---

# Securing the server

The API reads YAML files and rows from the server's disk, writes checkpoints, and starts training. On a machine only
you can reach, that is convenient; on one others can reach, it is a problem. Two settings close it:

- **`API_TOKEN`**: when set, every `/api` route requires `Authorization: Bearer <token>`. `/health` stays open so a
  container healthcheck can call it.
- **`API_PATHS`**: a colon-separated list of directories the API may read configs and rows from and write output to.
  Every path a request names (the config file, `output_dir`, model directories, local row files) is resolved,
  following `..` and symlinks, and must land inside one of them; anything else is refused with 403. The default is
  the working directory.

Both are empty or local by default, so a single-user setup behaves as before. Set both on any server with a network
interface other people can see. The CLI is not affected by either.

Two more habits that matter: keep `HF_TOKEN` and API keys in `environment/.env` (ignored by git) or the environment,
never in a config; and pin `model.revision` to a commit so a run today and a run next month load the same weights.

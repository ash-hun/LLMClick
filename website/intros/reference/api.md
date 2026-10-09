---
title: HTTP API
description: Every endpoint of the server, generated from its OpenAPI description.
---

# HTTP API

The server is a FastAPI application; `llmclick serve` starts it, Swagger lives at `/docs`. All routes under `/api`
take `Authorization: Bearer <API_TOKEN>` when that setting is configured. Errors are `{"detail": "..."}` with 401
(no or wrong token), 403 (a path outside `API_PATHS`), 404 (missing config, job or report), 409 (cancelling a
finished job) or 422 (a request or config that does not fit the schema).

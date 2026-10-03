# Deploying the JobBuddy API

The API is a FastAPI service (`src/run_service.py`). This guide deploys it to
Render's free plan and checks it with a signed-out smoke test.

## What gets deployed

| Piece | Where | Notes |
| --- | --- | --- |
| FastAPI service (`/invoke`, `/stream`, `/history`, `/roadmap`) | Render free web service | 512 MB RAM, sleeps after 15 minutes without traffic |
| Model calls | Groq free plan | `GROQ_API_KEY` |
| Traces | LangSmith | `LANGCHAIN_API_KEY`, optional |
| Conversation state | SQLite at `/tmp/checkpoints.db` | lost when the instance sleeps or redeploys, see "Durable threads" |

The Next.js frontend in `frontend/` is not part of this deployment.

## Deploy to Render

1. Push the branch that contains `render.yaml` to GitHub.
2. In the Render dashboard choose **New > Blueprint** and connect the repo. Pick the branch.
3. Render reads `render.yaml` and asks for the two secrets marked `sync: false`:
   - `GROQ_API_KEY`: your Groq key
   - `LANGCHAIN_API_KEY`: your LangSmith key (leave empty to skip tracing)
4. Click **Apply**. The first build installs the pinned packages in `requirements.txt`
   and takes several minutes.
5. When the service is live, copy its URL, for example `https://jobbuddy-api.onrender.com`.

Without a Blueprint, create a **Web Service** by hand with the same values:
runtime Python 3, build command `pip install -r requirements.txt`, start command
`python src/run_service.py`, health check path `/health`, instance type Free,
plus the environment variables listed in `render.yaml`.

Render sets `PORT`; `run_service.py` reads it and binds to `0.0.0.0`.

### Why requirements.txt

`uv.lock` is git-ignored in this template, so a host that builds from the repo cannot
see it. `requirements.txt` is the same resolution exported with exact versions, so
the deployed build matches the tested one. Regenerate it after changing dependencies
(the command is at the top of the file). `tests/xbuddy/test_deploy_smoke.py` fails if
a dependency in `pyproject.toml` is missing from it.

## Signed-out smoke test

"Signed out" means no `Authorization` header and no cookies: what a stranger with
only the URL can do. Run it from any machine, with no `.env` needed:

```bash
uv run python src/smoke_test.py https://jobbuddy-api.onrender.com --expect open
```

It waits for a sleeping instance (up to 2 minutes), then checks `/health`, runs one
turn through `/invoke`, reads it back through `/history`, runs a second turn through
`/stream`, and confirms `/roadmap` reports "not ready". It fails if the model only
returned the fallback reply, which means the model key is missing or wrong on the server.
The exit code is 0 on success and 1 on failure.

The same checks by hand, in a private browser window (signed out of Render and GitHub):

- `https://<your-app>/health` shows `{"status":"ok"}`
- `https://<your-app>/docs` opens the interactive API page; **POST /invoke** with
  `{"message": "I'm a backend developer with 3 years of Python", "user_id": 1}` returns
  a reply and `custom_data.section.id` equal to `background`

## Open or protected

The server has two modes, chosen by one environment variable:

| Mode | `AUTH_SECRET` | Signed-out caller | Smoke test |
| --- | --- | --- | --- |
| open (default in `render.yaml`) | not set | can chat | `--expect open` runs a full conversation |
| protected | set | gets 401 on every API route, `/health` stays public | `--expect protected` checks the 401s; add `--token` to also run the conversation |

The demo is open so reviewers can try it without credentials. That means anyone with
the URL can spend the Groq free quota, and anyone who knows a thread id can read that
thread. Thread ids are random UUIDs and the app stores no contact details, salary, or
resume text, but a real launch should set `AUTH_SECRET` (or add user accounts) first.

To lock the deployment: add `AUTH_SECRET` in the Render dashboard, redeploy, then run

```bash
uv run python src/smoke_test.py https://jobbuddy-api.onrender.com --expect protected
uv run python src/smoke_test.py https://jobbuddy-api.onrender.com --token "$AUTH_SECRET"
```

## Durable threads

On the free plan the filesystem is temporary, so SQLite checkpoints disappear when
the instance sleeps, restarts, or redeploys. A returning user would start over.
That is acceptable for a smoke test, not for real users.

For durable threads, point the checkpointer at Postgres (for example the Supabase
project already used for the section mirror) by setting these on Render:

```
DATABASE_TYPE=postgres
POSTGRES_HOST=<session pooler host>
POSTGRES_PORT=5432
POSTGRES_USER=<pooler user>
POSTGRES_PASSWORD=<database password>
POSTGRES_DB=postgres
```

Use Supabase's **session pooler** connection details: Render has no outbound IPv6,
and Supabase's direct connection is IPv6 only. The service connects with
`sslmode=require`.

## Memory

A free instance has 512 MB. Provider SDKs are imported only when used
(`src/core/llm.py`), which brings startup memory from about 430 MB down to about
125 MB. `tests/xbuddy/test_deploy_smoke.py` guards this.

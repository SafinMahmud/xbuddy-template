# Deploying the JobBuddy API

> **Demo only.** The deployment described here has no sign-in. Anyone with the URL
> can use it (and spend the Groq free quota), and anyone who knows a thread id can
> read that thread. Do not enter personal or sensitive information, and do not send
> real users to it. Before real use: set `AUTH_SECRET` (see "Open or protected"),
> add rate limits, and keep the Postgres durability check below passing.
>
> Put this notice next to the URL wherever you share it. The API repeats it on its
> own `/docs` page, and `/health` reports `"access": "open"`.

The API is a FastAPI service (`src/run_service.py`). This guide deploys it to
Render's free plan and checks it with a signed-out smoke test.

## What gets deployed

| Piece | Where | Notes |
| --- | --- | --- |
| FastAPI service (`/invoke`, `/stream`, `/history`, `/roadmap`) | Render free web service | 512 MB RAM, sleeps after 15 minutes without traffic |
| Conversation state (LangGraph checkpoints) | Render free Postgres | private network only; expires 30 days after creation |
| Model calls | Groq free plan | `GROQ_API_KEY` |
| Traces | LangSmith | `LANGCHAIN_API_KEY`, optional |

The Next.js frontend in `frontend/` is not part of this deployment.

## Deploy to Render

1. Push the branch that contains `render.yaml` to GitHub.
2. In the Render dashboard choose **New > Blueprint** and connect the repo. Pick the branch.
3. Render reads `render.yaml` and asks for the two secrets marked `sync: false`:
   - `GROQ_API_KEY`: your Groq key
   - `LANGCHAIN_API_KEY`: your LangSmith key (leave empty to skip tracing)
4. Click **Apply**. Render creates the database and the service. The first build
   installs the pinned packages in `requirements.txt` and takes several minutes.
5. When the service is live, copy its URL, for example `https://jobbuddy-api.onrender.com`.

After changing `render.yaml` later, open the Blueprint in the dashboard and click
**Manual sync** (or enable auto sync) so Render applies the change.

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
The report also prints the storage type and the thread id it used. The exit code is
0 on success and 1 on failure.

The same checks by hand, in a private browser window (signed out of Render and GitHub):

- `https://<your-app>/health` shows `"status":"ok"` plus the access mode and storage
- `https://<your-app>/docs` opens the interactive API page; **POST /invoke** with
  `{"message": "I'm a backend developer with 3 years of Python", "user_id": 1}` returns
  a reply and `custom_data.section.id` equal to `background`

## Durability check: does a thread survive a restart?

A returning user must find their conversation where they left it, so this is part of
the acceptance path, not an extra.

```bash
# 1. Run the smoke test and note the "Thread:" line
uv run python src/smoke_test.py https://jobbuddy-api.onrender.com --expect open

# 2. Restart the service: Render dashboard > jobbuddy-api > Manual Deploy > Restart service
#    (or deploy again). Wait until it is live.

# 3. Ask for the same thread back
uv run python src/smoke_test.py https://jobbuddy-api.onrender.com --resume <thread id>
```

Step 3 passes only if both earlier turns are still in `/history`, section progress is
intact, and a new turn is added to the same thread. With Postgres it passes. With
SQLite on a temporary filesystem it fails with "thread is gone", which is the honest
result: that storage loses every thread when the instance sleeps or redeploys.

`/health` shows which one a deployment uses:
`"storage": {"type": "postgres", "durable": true}`.

The same guarantee is tested against a real database before deploying:

```bash
TEST_POSTGRES_URL=postgresql://user:password@host:5432/dbname \
    uv run pytest tests/xbuddy/test_postgres_live.py -q
```

These tests are skipped when `TEST_POSTGRES_URL` is not set.

## Storage options

| Setting | Durable | Use |
| --- | --- | --- |
| `DATABASE_TYPE=sqlite` (local default) | only while the file exists | local development |
| `DATABASE_TYPE=postgres` + `POSTGRES_URL` (the Blueprint) | yes | deployed service |

- **Render free Postgres** is wired by `render.yaml`. It is reachable only from the
  service (`ipAllowList: []`). It **expires 30 days after creation**; when it does, the
  service cannot start. Upgrade the database, or point `POSTGRES_URL` at another one.
- **Supabase** (or any other Postgres): set `POSTGRES_URL` to its connection URL in the
  Render dashboard. For Supabase use the **session pooler** URL: Render has no outbound
  IPv6, and Supabase's direct connection is IPv6 only.
- The five separate settings (`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`,
  `POSTGRES_PASSWORD`, `POSTGRES_DB`) still work, with `POSTGRES_SSLMODE` (default `require`).
- The database must use **UTF8** encoding. The service refuses to start otherwise: on a
  SQL_ASCII database every turn after the first is silently written to the wrong thread.

LangGraph's checkpointer is the source of truth for a conversation. The Supabase
section mirror from PR 4 stays best-effort and is not needed for durability.

## Open or protected

The server has two modes, chosen by one environment variable:

| Mode | `AUTH_SECRET` | Signed-out caller | Smoke test |
| --- | --- | --- | --- |
| open (default in `render.yaml`) | not set | can chat | `--expect open` runs a full conversation |
| protected | set | gets 401 on every API route, `/health` stays public | `--expect protected` checks the 401s; add `--token` to also run the conversation |

The demo is open so reviewers can try it without credentials. That is a deliberate,
temporary choice with real costs: strangers can spend the Groq quota and a leaked
thread id exposes that thread. Thread ids are random UUIDs and the app is designed
not to store contact details, salary or resume text, but that is not a substitute
for authentication.

To lock the deployment: add `AUTH_SECRET` in the Render dashboard, redeploy, then run

```bash
uv run python src/smoke_test.py https://jobbuddy-api.onrender.com --expect protected
uv run python src/smoke_test.py https://jobbuddy-api.onrender.com --token "$AUTH_SECRET"
```

`AUTH_SECRET` is one shared token, which is enough to close a demo but not enough for
real users: they need per-user accounts, so one user cannot read another's thread,
and per-user rate limits.

## Memory

A free instance has 512 MB. Provider SDKs are imported only when used
(`src/core/llm.py`), which brings startup memory from about 430 MB down to about
125 MB. `tests/xbuddy/test_deploy_smoke.py` guards this.

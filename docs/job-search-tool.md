# Job search tool (Phase 8, tool use)

JobBuddy can look up real, current job openings in the middle of a conversation.
The model decides when: it calls the `search_jobs` tool when the user asks to see
openings, or says they have no posting to paste in the Skill Gap section.

## How a tool turn runs

```
generate_reply ── model asks for search_jobs ──> tools ── result ──> generate_reply ── reply ──> generate_decision
```

1. `generate_reply` gives the model the tool (in Target Role, Skill Gap and
   Application Strategy, and after the roadmap is delivered).
2. If the model answers with a tool call, the node stores it in `tool_scratch` and
   returns no reply. `route_after_reply` sends the graph to `tools`.
3. `tools` (`nodes/run_tools.py`) runs the call and adds the result to `tool_scratch`.
4. Back in `generate_reply`, the model writes the reply from the result. The scratch
   is emptied.

Tool calls and results never enter `messages`, so the chat transcript, `/history`
and `/stream` only ever contain what the user said and what JobBuddy answered.

## Setup

1. Create a free key at https://developer.adzuna.com/signup .
2. Local: add to `.env`

   ```
   ADZUNA_APP_ID=...
   ADZUNA_APP_KEY=...
   JOB_SEARCH_COUNTRY=ca
   ```

3. Render: add `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in the dashboard (the Blueprint
   lists them with `sync: false`, so they are never committed), then redeploy.

Try it: get to the Skill Gap section and say "I don't have any postings, can you find some?"

## What happens when it fails

| Failure | Behaviour |
| --- | --- |
| Timeout, connection error, HTTP 429 or 5xx | retried twice with a short backoff, then reported |
| Key missing or rejected, HTTP 4xx | reported at once, no retry |
| Model sends bad arguments or an unknown tool | error result for that call, other calls still run |
| Model keeps asking for searches | stopped after 2 searches in one turn |
| Provider rejects the tool call itself | the reply is generated again without tools |
| Model has no tool support | tools are skipped, replies work as before |

"Reported" means the tool returns `{"status": "error", "message": ...}` instead of
raising. The model tells the user in one sentence and carries on with the section,
for example by asking them to paste a posting. A failed search never fails the turn.

## Design choices

- **Results are leads, not evidence.** The tool returns title, company, location,
  date and link. Adzuna's description is a cut-off snippet, so it is dropped: the
  gap analysis still needs the full posting text pasted by the user, which keeps
  every requirement traceable to a real quote.
- **No salary fields.** JobBuddy gives no salary advice, and the roadmap validator
  rejects salary figures, so they are removed before the model sees the results.
- **Own tools node instead of the prebuilt `ToolNode`.** Same job, but it writes to
  `tool_scratch` and its failure rules are visible in 20 lines.

## Trace for the PR

With `LANGCHAIN_TRACING_V2=true`, a tool turn shows up in LangSmith as
`generate_reply → tools → search_jobs → generate_reply`. To show the failure path,
set a wrong `ADZUNA_APP_KEY`, ask for openings again, and share that trace too.

## Tests

`uv run pytest tests/xbuddy/test_job_search.py -q` (no network, no keys needed).

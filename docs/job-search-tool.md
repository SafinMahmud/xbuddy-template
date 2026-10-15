# Job search tool (Phase 8, tool use)

JobBuddy can look up recent job openings in the middle of a conversation. The
model decides when: it calls the `search_jobs` tool when the user asks to see
openings, or says they have no posting to paste in the Skill Gap section.

The openings come from [job-tracker](https://github.com/SafinMahmud/job-tracker), a
separate project that scrapes company career boards (Greenhouse, Lever, Ashby,
Workday) every day and keeps software roles in Ontario and remote Canada. Its
`jobs` table is public-read, so JobBuddy reads it through Supabase's REST API with
the public anon key. No paid plan, no call quota, no change to job-tracker.

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

1. Find job-tracker's public anon key: Supabase > the job-tracker project >
   Project Settings > API Keys > `anon` `public`.
2. Local: add to `.env`

   ```
   JOB_TRACKER_SUPABASE_URL=https://sptzvuolqbqtmmiyowgy.supabase.co
   JOB_TRACKER_SUPABASE_KEY=...
   ```

3. Render: add the same two variables in the dashboard (the Blueprint lists them
   with `sync: false`, so they are never committed), then redeploy.

Try it: get to the Skill Gap section and say "I don't have any postings, can you find some?"

## What happens when it fails

| Failure | Behaviour |
| --- | --- |
| Timeout, connection error, HTTP 429 or 5xx | retried twice with a short backoff, then reported |
| Key missing or rejected, HTTP 4xx | reported at once, no retry |
| Query has no real words ("jobs in the") | refused before any request |
| Nothing matches | `no_results`, with what the search covers, so the model can say so |
| Model sends bad arguments or an unknown tool | error result for that call, other calls still run |
| Model keeps asking for searches | stopped after 2 searches in one turn |
| Provider rejects the tool call itself | the reply is generated again without tools |
| Model has no tool support | tools are skipped, replies work as before |

"Reported" means the tool returns `{"status": "error", "message": ...}` instead of
raising. The model tells the user in one sentence and carries on with the section,
for example by asking them to paste a posting. A failed search never fails the turn.

## Design choices

- **My own job data instead of a paid job API.** Adzuna moved new accounts to a
  25-call trial and a monthly fee, and the free alternatives either barely cover
  Toronto or allow a few calls a day. job-tracker already collects exactly the
  roles JobBuddy's users target, straight from company career boards.
- **Results are leads, not evidence.** The tool returns title, company, location,
  date, board and link. The stored description is never fetched: the gap analysis
  still needs the full posting pasted by the user, which keeps every requirement
  traceable to a real quote.
- **The model never writes the filter.** Three layers: search words are cut down to
  letters, digits, `+` and `#`; each value is double-quoted with quotes and
  backslashes escaped (PostgREST's escaping for logic trees); and httpx
  percent-encodes the query, so `+` and `#` arrive as `%2B` and `%23`. Tests feed
  hostile text (quote breakouts, extra clauses, extra query parameters, SQL-style
  input) and check that only the one allowed filter shape ever comes out.
- **Read-only by design.** The anon key can only read the public `jobs` table; the
  row-level security policies in job-tracker keep every other table closed.
- **Own tools node instead of the prebuilt `ToolNode`.** Same job, but it writes to
  `tool_scratch` and its failure rules are visible in 20 lines.

## Trace for the PR

With `LANGCHAIN_TRACING_V2=true`, a tool turn shows up in LangSmith as
`generate_reply → tools → search_jobs → generate_reply`. To show the failure path,
set a wrong `JOB_TRACKER_SUPABASE_KEY`, ask for openings again, and share that trace too.

## Tests

`uv run pytest tests/xbuddy/test_job_search.py -q` (no network, no keys needed).

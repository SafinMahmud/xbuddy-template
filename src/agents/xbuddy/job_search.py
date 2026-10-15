"""Job search tool: recent openings from the job-tracker database.

The data comes from job-tracker (github.com/SafinMahmud/job-tracker), a separate
project that scrapes company career boards (Greenhouse, Lever, Ashby, Workday)
every day and keeps software roles in Ontario and remote Canada. Its `jobs`
table is public-read (row-level security policy `jobs_select_public`), so this
tool reads it through Supabase's REST API with the project's public anon key:
no paid plan, no call quota, nothing to change in job-tracker.

The model can call `search_jobs` mid-conversation (see nodes/generate_reply.py
and nodes/run_tools.py). Design rules:

  - The tool never raises. Every outcome is a small JSON string with a "status"
    of "ok", "no_results" or "error", so a failed search becomes something the
    model can explain to the user instead of a crashed turn.
  - Transient failures (timeout, connection error, 429, 5xx) are retried with a
    short backoff. Permanent ones (missing or rejected key, bad request) are not.
  - Results carry title, company, location, date, board and link only. The stored
    description stays out: the gap analysis needs the full posting from the user.
  - Search words are reduced to letters and digits before they reach the query,
    so model output can never change the shape of the database filter.
"""

import asyncio
import json
import logging
import re

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from core.settings import settings

logger = logging.getLogger(__name__)

SOURCE = "job-tracker (company career boards: Greenhouse, Lever, Ashby, Workday)"
COVERAGE = "software roles in Ontario and remote Canada"
MAX_RESULTS = 5
MAX_WORDS = 4
MAX_ATTEMPTS = 3  # first try + two retries
BACKOFF_SECONDS = (0.5, 1.5)  # wait before retry 1 and retry 2
TIMEOUT = httpx.Timeout(8.0, connect=4.0)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
COLUMNS = "title,company,location,url,ats_platform,date_discovered"
# Words that say "a job" rather than which job, so they would only narrow the search.
FILLER_WORDS = frozenset(
    {"job", "jobs", "role", "roles", "position", "positions", "opening", "openings",
     "in", "at", "for", "and", "or", "the", "a", "an", "near", "remote"}
)  # fmt: skip

# Tests replace this with an httpx.MockTransport, so no test touches the network.
_TRANSPORT: httpx.AsyncBaseTransport | None = None


class JobPosting(BaseModel):
    title: str
    company: str | None = None
    location: str | None = None
    posted: str | None = None  # YYYY-MM-DD, when job-tracker first saw it
    board: str | None = None  # greenhouse, lever, ashby, workday
    url: str


class JobSearchError(Exception):
    """A search that failed. `str(exc)` is safe to show to the user."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def is_configured() -> bool:
    return bool(settings.JOB_TRACKER_SUPABASE_URL and settings.JOB_TRACKER_SUPABASE_KEY)


def search_words(query: str) -> list[str]:
    """The words a title or company must contain. Letters, digits, + and # only."""
    words = re.findall(r"[a-z0-9+#]+", query.lower())
    return [w for w in words if w not in FILLER_WORDS][:MAX_WORDS]


def location_term(location: str) -> str:
    """'Toronto, ON' -> 'toronto'. The part before the first comma, letters only."""
    first = location.split(",")[0]
    return " ".join(re.findall(r"[a-z]+", first.lower()))


def build_params(words: list[str], location: str, limit: int) -> dict[str, str]:
    """PostgREST filter: every word in the title or company, newest first."""
    params = {"select": COLUMNS, "order": "date_discovered.desc", "limit": str(limit)}
    if words:
        clauses = ",".join(f"or(title.ilike.*{w}*,company.ilike.*{w}*)" for w in words)
        params["and"] = f"({clauses})"
    if location:
        params["location"] = f"ilike.*{location}*"
    return params


def _parse(rows: list, limit: int) -> list[JobPosting]:
    postings = []
    for row in rows if isinstance(rows, list) else []:
        title, url = (row.get("title") or "").strip(), row.get("url")
        if not title or not url:
            continue  # a posting the user cannot open is not worth showing
        postings.append(
            JobPosting(
                title=title,
                company=row.get("company"),
                location=row.get("location"),
                posted=(row.get("date_discovered") or "")[:10] or None,
                board=row.get("ats_platform"),
                url=url,
            )
        )
    return postings[:limit]


async def _get_once(client: httpx.AsyncClient, url: str, params: dict, headers: dict) -> list:
    try:
        response = await client.get(url, params=params, headers=headers)
    except httpx.TimeoutException:
        raise JobSearchError("The job search service took too long to answer.", True) from None
    except httpx.HTTPError:
        raise JobSearchError("The job search service could not be reached.", True) from None

    if response.status_code in (401, 403):
        raise JobSearchError("The job search key was rejected.")
    if response.status_code in RETRYABLE_STATUS:
        raise JobSearchError(
            f"The job search service is busy (HTTP {response.status_code}).", True
        )
    if response.status_code >= 400:
        raise JobSearchError(f"The job search request was refused (HTTP {response.status_code}).")
    try:
        return response.json()
    except ValueError:
        raise JobSearchError("The job search service sent an unreadable answer.", True) from None


async def fetch_jobs(query: str, location: str = "", limit: int = MAX_RESULTS) -> list[JobPosting]:
    """Query job-tracker's jobs table, retrying transient failures. Raises JobSearchError."""
    if not is_configured():
        raise JobSearchError("Job search is not set up on this server.")
    words = search_words(query)
    if not words:
        raise JobSearchError("The search needs a job title or skill, for example 'backend'.")

    base = settings.JOB_TRACKER_SUPABASE_URL.rstrip("/")
    key = settings.JOB_TRACKER_SUPABASE_KEY.get_secret_value()
    headers = {"apikey": key, "Accept": "application/json"}
    params = build_params(words, location_term(location), limit)

    async with httpx.AsyncClient(timeout=TIMEOUT, transport=_TRANSPORT) as client:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                rows = await _get_once(client, f"{base}/rest/v1/jobs", params, headers)
                return _parse(rows, limit)
            except JobSearchError as exc:
                if not exc.retryable or attempt == MAX_ATTEMPTS:
                    raise
                logger.warning("job search attempt %d failed: %s", attempt, exc)
                await asyncio.sleep(BACKOFF_SECONDS[attempt - 1])
    raise JobSearchError("The job search did not finish.")  # unreachable, keeps types honest


class SearchJobsInput(BaseModel):
    """Arguments the model fills in. Plain strings only, so every provider accepts the schema."""

    query: str = Field(
        ..., min_length=2, max_length=80,
        description="One to three words from a job title or a skill, for example 'backend' "
        "or 'python developer'. Fewer words find more jobs.",
    )  # fmt: skip
    location: str = Field(
        "", max_length=60,
        description="A city such as 'Toronto', or 'remote'. Empty for all of Ontario and "
        "remote Canada.",
    )  # fmt: skip


@tool("search_jobs", args_schema=SearchJobsInput)
async def search_jobs(query: str, location: str = "") -> str:
    """Search recent software job openings in Ontario and remote Canada, newest first.

    The jobs come from company career boards (Greenhouse, Lever, Ashby, Workday).
    Returns JSON with "status": "ok" (with "results"), "no_results", or "error"
    (with a "message" to pass on to the user).
    """
    try:
        postings = await fetch_jobs(query, location)
    except JobSearchError as exc:
        logger.warning("search_jobs failed: %s", exc)
        return json.dumps({"status": "error", "message": str(exc)})
    except Exception:  # a tool bug must not take the turn down with it
        logger.exception("search_jobs crashed")
        return json.dumps({"status": "error", "message": "The job search failed unexpectedly."})
    if not postings:
        return json.dumps(
            {"status": "no_results", "query": query, "location": location, "covers": COVERAGE}
        )
    return json.dumps(
        {
            "status": "ok",
            "source": SOURCE,
            "covers": COVERAGE,
            "results": [p.model_dump(exclude_none=True) for p in postings],
        }
    )


TOOLS = [search_jobs]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

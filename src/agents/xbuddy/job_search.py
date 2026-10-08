"""Job search tool: real, current openings from the Adzuna API.

The model can call `search_jobs` mid-conversation (see nodes/generate_reply.py and
nodes/tools.py). Design rules:

  - The tool never raises. Every outcome is a small JSON string with a "status"
    of "ok", "no_results" or "error", so a failed search becomes something the
    model can explain to the user instead of a crashed turn.
  - Transient failures (timeout, connection error, 429, 5xx) are retried with a
    short backoff. Permanent ones (missing or rejected key, bad request) are not.
  - Results carry title, company, location, date and link only. Salary fields are
    dropped on purpose (JobBuddy gives no salary advice) and so are descriptions:
    Adzuna returns a cut-off snippet, which must not be used as posting evidence.

Free key: https://developer.adzuna.com/signup (ADZUNA_APP_ID, ADZUNA_APP_KEY).
"""

import asyncio
import json
import logging

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from core.settings import settings

logger = logging.getLogger(__name__)

ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"
SOURCE = "Adzuna"
MAX_RESULTS = 5
MAX_ATTEMPTS = 3  # first try + two retries
BACKOFF_SECONDS = (0.5, 1.5)  # wait before retry 1 and retry 2
TIMEOUT = httpx.Timeout(8.0, connect=4.0)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

COUNTRIES = frozenset({
    "at", "au", "be", "br", "ca", "ch", "de", "es", "fr", "gb",
    "in", "it", "mx", "nl", "nz", "pl", "sg", "us", "za",
})  # fmt: skip

# Tests replace this with an httpx.MockTransport, so no test touches the network.
_TRANSPORT: httpx.AsyncBaseTransport | None = None


class JobPosting(BaseModel):
    title: str
    company: str | None = None
    location: str | None = None
    posted: str | None = None  # YYYY-MM-DD
    url: str


class JobSearchError(Exception):
    """A search that failed. `str(exc)` is safe to show to the user."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def is_configured() -> bool:
    return bool(settings.ADZUNA_APP_ID and settings.ADZUNA_APP_KEY)


def _parse(payload: dict, limit: int) -> list[JobPosting]:
    postings = []
    for item in payload.get("results") or []:
        title, url = (item.get("title") or "").strip(), item.get("redirect_url")
        if not title or not url:
            continue  # a posting the user cannot open is not worth showing
        postings.append(
            JobPosting(
                title=title,
                company=(item.get("company") or {}).get("display_name"),
                location=(item.get("location") or {}).get("display_name"),
                posted=(item.get("created") or "")[:10] or None,
                url=url,
            )
        )
    return postings[:limit]


async def _get_once(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    try:
        response = await client.get(url, params=params)
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


async def fetch_jobs(
    query: str, location: str | None, country: str, limit: int = MAX_RESULTS
) -> list[JobPosting]:
    """Call Adzuna, retrying transient failures. Raises JobSearchError."""
    if not is_configured():
        raise JobSearchError("Job search is not set up on this server.")
    params = {
        "app_id": settings.ADZUNA_APP_ID.get_secret_value(),
        "app_key": settings.ADZUNA_APP_KEY.get_secret_value(),
        "what": query,
        "results_per_page": limit,
        "sort_by": "date",
        "content-type": "application/json",
    }
    if location:
        params["where"] = location
    url = ADZUNA_URL.format(country=country)

    async with httpx.AsyncClient(timeout=TIMEOUT, transport=_TRANSPORT) as client:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return _parse(await _get_once(client, url, params), limit)
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
        description="Job title or keywords, for example 'backend developer python'.",
    )  # fmt: skip
    location: str = Field(
        "", max_length=60,
        description="City or region, for example 'Toronto'. Empty for the whole country.",
    )  # fmt: skip
    country: str = Field(
        "", description="Two-letter country code such as 'ca', 'us' or 'gb'. Empty for the default."
    )


@tool("search_jobs", args_schema=SearchJobsInput)
async def search_jobs(query: str, location: str = "", country: str = "") -> str:
    """Search real, current job openings by title or keywords and optional location.

    Returns JSON with "status": "ok" (with "results"), "no_results", or "error"
    (with a "message" to pass on to the user).
    """
    # An unknown country (the model wrote "Canada", say) falls back to the default
    # instead of failing the call.
    country = country.strip().lower()
    if country not in COUNTRIES:
        country = settings.JOB_SEARCH_COUNTRY
    location = location.strip() or None
    try:
        postings = await fetch_jobs(query, location, country)
    except JobSearchError as exc:
        logger.warning("search_jobs failed: %s", exc)
        return json.dumps({"status": "error", "message": str(exc)})
    except Exception:  # a tool bug must not take the turn down with it
        logger.exception("search_jobs crashed")
        return json.dumps({"status": "error", "message": "The job search failed unexpectedly."})
    if not postings:
        return json.dumps({"status": "no_results", "query": query, "location": location})
    return json.dumps(
        {
            "status": "ok",
            "source": SOURCE,
            "country": country,
            "results": [p.model_dump(exclude_none=True) for p in postings],
        }
    )


TOOLS = [search_jobs]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

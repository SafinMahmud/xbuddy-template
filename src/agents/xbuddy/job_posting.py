"""Posting reader tool: the full text of a job the user picked from a search.

`search_jobs` lists openings from job-tracker. When the user picks one ("use
number 2", or pastes one of those links), the model calls `read_job_posting`
with the link, and JobBuddy reads the posting itself instead of asking the user
to copy it:

  1. Look the link up in job-tracker's `jobs` table. Greenhouse, Lever and Ashby
     rows carry the full description.
  2. Workday rows are stored without one (the Workday listing API has no
     descriptions), so the text comes from Workday's job detail endpoint, the
     same API job-tracker's scraper uses for listings.
  3. Anything else (LinkedIn, a company page job-tracker never saw) is not
     fetched. The user is asked to paste the text, in one sentence.

Only links already in job-tracker are ever fetched, and the Workday host is
rebuilt from a strict pattern, so a pasted link can never make the server call
an arbitrary address. Like search_jobs, the tool never raises.
"""

import html
import json
import logging
import re
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .job_search import JobSearchError, get_json, tracker_request

logger = logging.getLogger(__name__)

MAX_TEXT = 6000  # characters of posting text handed to the model
WORKDAY_SERVICE = "Workday"
BLOCKED_HOSTS = ("linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com")
# https://{tenant}.wdN.myworkdayjobs.com/{optional locale}/{site}/job/...
WORKDAY_URL = re.compile(
    r"https://(?P<tenant>[a-z0-9-]+)\.(?P<wd>wd\d{1,2})\.myworkdayjobs\.com"
    r"(?:/[a-z]{2}-[A-Z]{2})?/(?P<site>[A-Za-z0-9_-]+)(?P<path>/job/[^?#\s]+)"
)
WORKDAY_HEADERS = {
    "Accept": "application/json",
    "Accept-Language": "en-US",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
}


def _result(status: str, message: str | None = None, **extra) -> str:
    data = {"status": status, **({"message": message} if message else {}), **extra}
    return json.dumps(data)


def html_to_text(raw: str) -> str:
    """Tags out, entities decoded, one blank line between blocks at most."""
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6])>", "\n", raw or "")
    text = re.sub(r"(?i)<li[^>]*>", "- ", text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_link(link: str) -> str | None:
    """The link as an absolute http(s) URL, or None if it is not one."""
    link = link.strip().strip("<>").strip()
    parts = urlsplit(link)
    if parts.scheme not in ("http", "https") or not parts.netloc or len(link) > 500:
        return None
    return link


def _blocked(link: str) -> bool:
    host = urlsplit(link).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in BLOCKED_HOSTS)


async def _lookup(link: str) -> dict | None:
    """The job-tracker row for this link. Retries once without the query string."""
    url, headers = tracker_request()
    columns = "title,company,location,url,ats_platform,description"
    candidates = [link]
    if "?" in link:
        candidates.append(link.split("?", 1)[0])
    for candidate in candidates:
        params = {"select": columns, "url": f"eq.{candidate}", "limit": "1"}
        rows = await get_json(url, params, headers)
        if isinstance(rows, list) and rows:
            return rows[0]
    return None


async def _workday_text(link: str) -> str:
    """Description from Workday's job detail endpoint. Raises JobSearchError."""
    match = WORKDAY_URL.fullmatch(link.split("?", 1)[0])
    if match is None:
        raise JobSearchError("This Workday link has an unexpected format.")
    tenant, wd, site, path = match.group("tenant", "wd", "site", "path")
    base = f"https://{tenant}.{wd}.myworkdayjobs.com"
    detail = await get_json(
        f"{base}/wday/cxs/{tenant}/{site}{path}",
        {},
        {**WORKDAY_HEADERS, "Referer": f"{base}/en-US/{site}"},
        service=WORKDAY_SERVICE,
    )
    info = detail.get("jobPostingInfo") if isinstance(detail, dict) else None
    return html_to_text((info or {}).get("jobDescription") or "")


class ReadJobPostingInput(BaseModel):
    link: str = Field(
        ...,
        description="The posting's link, exactly as it appeared in the search results "
        "or as the user pasted it.",
    )


@tool("read_job_posting", args_schema=ReadJobPostingInput)
async def read_job_posting(link: str) -> str:
    """Read the full text of a job posting the user picked from the search results.

    Returns JSON with "status": "ok" (with "posting": label, url, location, text),
    or "unsupported", "not_found", "no_text" or "error" (with a "message" to pass
    on to the user, who can paste the text instead).
    """
    url = clean_link(link)
    if url is None:
        return _result("error", "That doesn't look like a link. Please paste the posting text.")
    if _blocked(url):
        return _result(
            "unsupported",
            "Job sites like LinkedIn can't be read automatically. Please paste the posting text.",
        )
    try:
        row = await _lookup(url)
        if row is None:
            return _result(
                "not_found",
                "I can only read jobs from my own search. Please paste the posting text.",
            )
        text = html_to_text(row.get("description") or "")
        if not text and row.get("ats_platform") == "workday":
            text = await _workday_text(row.get("url") or url)
    except JobSearchError as exc:
        logger.warning("read_job_posting failed: %s", exc)
        return _result("no_text", f"{exc} Please open the link and paste the posting text.")
    except Exception:  # a tool bug must not take the turn down with it
        logger.exception("read_job_posting crashed")
        return _result("error", "Reading the posting failed. Please paste the posting text.")

    if not text:
        return _result(
            "no_text",
            "I found this job but its description isn't available. Please paste the text.",
        )
    title, company = row.get("title") or "Job", row.get("company") or "Unknown company"
    return _result(
        "ok",
        posting={
            "label": f"{company} - {title}",
            "url": row.get("url") or url,
            "location": row.get("location"),
            "text": text[:MAX_TEXT],
            "truncated": len(text) > MAX_TEXT,
        },
    )


MAX_POSTINGS = 3  # postings kept for the Skill Gap section, most recent last


def remember_postings(kept: list[dict], new: list[dict]) -> list[dict]:
    """Kept postings plus new ones, one per link, at most MAX_POSTINGS (newest kept)."""
    by_url = {p["url"]: p for p in kept}
    for posting in new:
        by_url.pop(posting["url"], None)
        by_url[posting["url"]] = posting
    return list(by_url.values())[-MAX_POSTINGS:]


def postings_block(postings: list[dict]) -> str:
    """The chosen postings as prompt text, for the Skill Gap reply and extraction."""
    heading = (
        "POSTINGS THE USER CHOSE (JobBuddy read them from company career boards). "
        "Treat them exactly like postings the user pasted, and use these labels:"
    )
    parts = [heading]
    for posting in postings:
        parts.append(f"=== {posting['label']}\n{posting['url']}\n{posting['text']}")
    return "\n\n".join(parts)


PASTED_POSTING_MIN_CHARS = 300  # a user message this long in Skill Gap counts as a posting


def section_has_posting(state: dict) -> bool:
    """True once the Skill Gap section has a real posting to work from.

    Either JobBuddy read one (fetched_postings), or the user pasted one: a user
    message in the section's memory at least PASTED_POSTING_MIN_CHARS long. A
    heuristic, but it is the same text extraction reads, so a section can't be
    confirmed with nothing for extraction to quote.
    """
    if state.get("fetched_postings"):
        return True
    return any(
        isinstance(m, HumanMessage) and len(str(m.content)) >= PASTED_POSTING_MIN_CHARS
        for m in state.get("short_memory") or []
    )

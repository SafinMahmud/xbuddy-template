"""Memory updater node — saves section progress and detects completion.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/memory_updater.py

After each decision:
  1. should_save_content -> store section_summary as the UNCONFIRMED draft
     (SectionState.content).
  2. next -> the user confirmed: mark the section done, store the confirmed
     summary, and extract structured JobBuddyData for that section. Fields the
     decision claimed as covered but extraction could not find are recorded in
     SectionState.unverified_fields, so the model's claim is never taken as proof.
  3. If no unfinished section remains, set should_generate_final_output so the
     graph goes to `implementation` instead of the router.
  4. Recovery: if extraction failed or was incomplete for a confirmed section,
     retry it on later turns (one section per turn, at most
     MAX_EXTRACTION_ATTEMPTS in total). If it still cannot be verified, the
     section's unverified_fields stay visible to the reply prompt, and the user
     can repair it by correcting that section, which re-runs extraction.
  5. Mirror changed sections to Supabase when configured.

Storage boundary: the LangGraph checkpointer (SQLite locally, Postgres in
production) is the authoritative store for all of this state. Supabase is a
best-effort mirror for the frontend; a failed mirror write is recorded and
never changes what the graph knows.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..enums import RouterDirective, SectionID, SectionStatus
from ..extraction import extract_section_data, structured_fields, unverified_fields
from ..models import JobBuddyData, SectionContent, SectionState, XBuddyState
from ..persistence import get_section_store
from ..prompts import get_next_unfinished_section

logger = logging.getLogger(__name__)

MAX_EXTRACTION_ATTEMPTS = 3  # one at confirmation, then up to two retries on later turns
RETRY_HISTORY_LIMIT = 40  # messages given to a retry


def to_tiptap(text: str) -> dict[str, Any]:
    """Plain text as Tiptap JSON (one paragraph per line) for the frontend editor."""
    paragraphs = [
        {"type": "paragraph", "content": [{"type": "text", "text": line}]}
        for line in text.splitlines()
        if line.strip()
    ]
    return {"type": "doc", "content": paragraphs}


def _record_error(state: XBuddyState, updates: dict[str, Any], message: str) -> None:
    updates["error_count"] = updates.get("error_count", state.get("error_count", 0)) + 1
    updates["last_error"] = message


async def _extract(
    section: SectionState,
    conversation: list,
    user_data: JobBuddyData,
    state: XBuddyState,
    updates: dict[str, Any],
    config: RunnableConfig,
) -> tuple[SectionState, JobBuddyData]:
    """Run extraction for a confirmed section and record what could not be verified.

    covered_fields was the model's claim; extraction is the check on real data.
    A failure never blocks the conversation: the fields stay unverified and the
    section is retried on later turns (see _retry_one_unverified_section).
    """
    sid = section.section_id
    try:
        user_data = await extract_section_data(
            sid, conversation, section.confirmed_summary or "", user_data, config
        )
        unverified = unverified_fields(sid, user_data)
    except Exception as exc:  # noqa: BLE001 - extraction must not block the conversation
        logger.warning("extraction failed for %s: %s", sid.value, exc)
        _record_error(state, updates, f"extraction[{sid.value}]: {exc}")
        unverified = structured_fields(sid)  # nothing could be verified
    if unverified:
        logger.warning("%s has unverified fields: %s", sid.value, unverified)
    section = section.model_copy(
        update={
            "unverified_fields": unverified,
            "extraction_attempts": section.extraction_attempts + 1,
        }
    )
    return section, user_data


def _needs_retry(section: SectionState) -> bool:
    return (
        section.status == SectionStatus.DONE
        and bool(section.unverified_fields)
        and section.extraction_attempts < MAX_EXTRACTION_ATTEMPTS
    )


async def memory_updater_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Update the current section from the decision, persist it, check completion."""
    output = state.get("agent_output")
    if output is None:
        return {}

    current: SectionID = state["current_section"]
    states = dict(state["section_states"])
    section = states.get(current.value) or SectionState(section_id=current)
    summary = (output.section_summary or "").strip()
    user_data: JobBuddyData = state["user_data"]
    updates: dict[str, Any] = {}
    changed: list[SectionState] = []

    before = section
    if output.should_save_content and summary:
        section = section.model_copy(
            update={"content": SectionContent(content=to_tiptap(summary), plain_text=summary)}
        )
    if output.is_satisfied is not None:
        section = section.model_copy(
            update={"satisfaction_status": "satisfied" if output.is_satisfied else "needs_improvement"}
        )

    confirmed_now = output.router_directive == RouterDirective.NEXT.value
    if confirmed_now:
        confirmed = summary or (section.content.plain_text if section.content else "")
        section = section.model_copy(
            update={
                "confirmed_summary": confirmed or None,
                "status": SectionStatus.DONE,
                "extraction_attempts": 0,  # a new confirmation starts a fresh attempt budget
            }
        )
        section, user_data = await _extract(
            section, list(state.get("short_memory") or []), user_data, state, updates, config
        )

    if section != before:
        states[current.value] = section
        changed.append(section)

    # Recovery: a transient failure at confirmation time must not leave a confirmed
    # section without structured data. Retry one such section per turn, a bounded
    # number of times, from the full conversation (short memory has moved on).
    if not confirmed_now:
        pending = next((s for s in states.values() if _needs_retry(s)), None)
        if pending is not None:
            retried, user_data = await _extract(
                pending, list(state.get("messages") or [])[-RETRY_HISTORY_LIMIT:],
                user_data, state, updates, config,
            )
            states[retried.section_id.value] = retried
            changed.append(retried)
            if not retried.unverified_fields:
                logger.info("recovered extraction for %s", retried.section_id.value)

    if confirmed_now and get_next_unfinished_section(states) is None:
        updates["should_generate_final_output"] = True
    if user_data is not state["user_data"]:
        updates["user_data"] = user_data
    if not changed:
        return updates

    updates["section_states"] = states
    store = get_section_store()
    if store is not None:
        for saved in changed:
            try:
                await store.save_section(state["user_id"], state["thread_id"], saved)
            except Exception as exc:  # noqa: BLE001 - checkpointer still holds the state
                logger.warning("Supabase save failed for %s: %s", saved.section_id.value, exc)
                _record_error(state, updates, f"persist[{saved.section_id.value}]: {exc}")
    return updates

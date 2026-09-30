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
  4. Persist the changed section to Supabase when configured, so the saved row
     matches what the user confirmed.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from ..enums import RouterDirective, SectionID, SectionStatus
from ..extraction import extract_section_data, structured_fields, unverified_fields
from ..models import SectionContent, SectionState, XBuddyState
from ..persistence import get_section_store
from ..prompts import get_next_unfinished_section

logger = logging.getLogger(__name__)


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


async def memory_updater_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Update the current section from the decision, persist it, check completion."""
    output = state.get("agent_output")
    if output is None:
        return {}

    current: SectionID = state["current_section"]
    states = dict(state["section_states"])
    section = states.get(current.value) or SectionState(section_id=current)
    before = section
    summary = (output.section_summary or "").strip()
    updates: dict[str, Any] = {}

    if output.should_save_content and summary:
        section = section.model_copy(
            update={"content": SectionContent(content=to_tiptap(summary), plain_text=summary)}
        )
    if output.is_satisfied is not None:
        section = section.model_copy(
            update={"satisfaction_status": "satisfied" if output.is_satisfied else "needs_improvement"}
        )

    if output.router_directive == RouterDirective.NEXT.value:
        confirmed = summary or (section.content.plain_text if section.content else "")
        section = section.model_copy(
            update={"confirmed_summary": confirmed or None, "status": SectionStatus.DONE}
        )
        # covered_fields was the model's claim; extraction is the check on real data.
        try:
            updates["user_data"] = await extract_section_data(
                current,
                list(state.get("short_memory") or []),
                confirmed,
                state["user_data"],
                config,
            )
            unverified = unverified_fields(current, updates["user_data"])
        except Exception as exc:  # noqa: BLE001 - extraction must not block the conversation
            logger.warning("extraction failed for %s: %s", current.value, exc)
            _record_error(state, updates, f"extraction[{current.value}]: {exc}")
            unverified = structured_fields(current)  # nothing could be verified
        if unverified:
            logger.warning("%s confirmed with unverified fields: %s", current.value, unverified)
        section = section.model_copy(update={"unverified_fields": unverified})

        states[current.value] = section
        if get_next_unfinished_section(states) is None:
            updates["should_generate_final_output"] = True

    if section == before:
        return updates

    states[current.value] = section
    updates["section_states"] = states

    store = get_section_store()
    if store is not None:
        try:
            await store.save_section(state["user_id"], state["thread_id"], section)
        except Exception as exc:  # noqa: BLE001 - checkpointer still holds the state
            logger.warning("Supabase save failed for %s: %s", current.value, exc)
            _record_error(state, updates, f"persist[{current.value}]: {exc}")
    return updates

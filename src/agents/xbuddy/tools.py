"""Agent tools: the section context loader.

Two different things can exist for a section, and they are kept separate:
  - SectionState.content: the in-progress, UNCONFIRMED draft. It lives in
    section_states, which the LangGraph checkpointer persists per thread.
    memory_updater (PR 4) populates it; Supabase persistence is also PR 4.
  - SectionState.confirmed_summary: what the user explicitly confirmed.

The packet's `draft` is only ever the real unconfirmed draft. A confirmed
summary is never passed off as a draft; for a reopened section it is shown
separately as the answer being corrected.
"""

from typing import Any

from langchain_core.tools import tool

from .enums import SectionID, SectionStatus
from .models import ContextPacket, SectionState
from .prompts import BASE_RULES, get_section_template


def _confirmed_recap(section_states: dict[str, SectionState], current: SectionID) -> str:
    """Summaries of done sections only, so the agent never recaps unconfirmed info."""
    lines = []
    for section_id in SectionID:
        state = section_states.get(section_id.value)
        if section_id == current or state is None:
            continue
        if state.status == SectionStatus.DONE and state.confirmed_summary:
            name = get_section_template(section_id).name
            lines.append(f"- {name}: {state.confirmed_summary}")
    return "\n".join(lines)


def build_context_packet(
    section_id: SectionID | str, section_states: dict[str, SectionState]
) -> ContextPacket:
    """Assemble the system prompt, status, draft, and checklist for one section."""
    template = get_section_template(section_id)
    sid = template.section_id
    state = section_states.get(sid.value) or SectionState(section_id=sid)

    parts = [BASE_RULES.strip(), template.system_prompt_template.strip()]
    recap = _confirmed_recap(section_states, sid)
    if recap:
        parts.append("CONFIRMED SO FAR (recap only this, do not re-ask):\n" + recap)

    if state.confirmed_summary:
        # Reopened section: the user is correcting an answer they confirmed earlier.
        parts.append(
            "PREVIOUSLY CONFIRMED ANSWER (the user is correcting it; change only what "
            "they ask, then re-confirm):\n" + state.confirmed_summary
        )

    draft = state.content  # unconfirmed work in progress, if any
    if draft is not None and draft.plain_text:
        parts.append(
            "UNCONFIRMED DRAFT FOR THIS SECTION (continue from it, do not start over):\n"
            + draft.plain_text
        )

    return ContextPacket(
        section_id=sid,
        status=state.status,
        system_prompt="\n\n---\n\n".join(parts),
        draft=draft,
        validation_rules={
            "required_fields": template.required_fields,
            "rules": [r.model_dump() for r in template.validation_rules],
        },
    )


@tool
async def get_context(
    user_id: int,
    thread_id: str,
    section_id: str,
    section_states: dict[str, Any] | None = None,
) -> dict:
    """Load the context packet for a section.

    Returns a dict with: section_id, status, system_prompt, draft, validation_rules.
    """
    states = {
        k: v if isinstance(v, SectionState) else SectionState.model_validate(v)
        for k, v in (section_states or {}).items()
    }
    return build_context_packet(section_id, states).model_dump()

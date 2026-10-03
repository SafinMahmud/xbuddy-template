"""Router node — section navigation and context loading.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/router.py

Directives:
  stay                 Keep the current section and rebuild its context from latest state.
  next                 The user confirmed the current section: mark it done and move to
                       the next unfinished section (or finish if none remain). If the
                       current section was never started, just start it.
  modify:<section_id>  The user corrected an earlier answer: reopen that section. Only
                       sections the user has already visited can be reopened.

After handling a directive the router resets it to "stay", so a new directive is
only acted on once. Like the other nodes, it returns only the keys it changes.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from ..enums import RouterDirective, SectionID, SectionStatus
from ..models import SectionState, XBuddyState
from ..prompts import get_next_unfinished_section
from ..tools import build_context_packet

logger = logging.getLogger(__name__)

MODIFY_PREFIX = f"{RouterDirective.MODIFY.value}:"


def _with_status(
    states: dict[str, SectionState], section_id: SectionID, status: SectionStatus
) -> dict[str, SectionState]:
    """Copy of section_states with one section's status changed (no in-place mutation)."""
    updated = dict(states)
    current = updated.get(section_id.value) or SectionState(section_id=section_id)
    updated[section_id.value] = current.model_copy(update={"status": status})
    return updated


def _enter(states: dict[str, SectionState], section_id: SectionID) -> dict[str, SectionState]:
    """Mark a section in progress when the user enters it (unless it's already done)."""
    state = states.get(section_id.value)
    if state is not None and state.status != SectionStatus.PENDING:
        return states
    return _with_status(states, section_id, SectionStatus.IN_PROGRESS)


def _switch_to(
    target: SectionID,
    current: SectionID,
    states: dict[str, SectionState],
    messages: list | None = None,
) -> dict[str, Any]:
    """Updates for moving into `target`: context, status, fresh short memory.

    Short memory restarts per section. It is seeded with the last AI message,
    which already asked the new section's opening question.
    """
    updates: dict[str, Any] = {
        "current_section": target,
        "section_states": states,
        "context_packet": build_context_packet(target, states),
        "router_directive": RouterDirective.STAY.value,
    }
    if target != current:
        last_ai = next((m for m in reversed(messages or []) if isinstance(m, AIMessage)), None)
        updates["short_memory"] = [last_ai] if last_ai is not None else []
    return updates


async def router_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Apply the router directive, pick the section, and load its context."""
    current: SectionID = state["current_section"]
    states: dict[str, SectionState] = state["section_states"]
    directive = str(state.get("router_directive") or RouterDirective.STAY.value).lower()

    updates: dict[str, Any] = {}
    msgs = state.get("messages", [])
    if msgs and isinstance(msgs[-1], HumanMessage):
        updates["awaiting_user_input"] = False

    # --- next: current section confirmed, advance -------------------------------
    if directive == RouterDirective.NEXT.value:
        current_state = states.get(current.value)
        if current_state is None or current_state.status == SectionStatus.PENDING:
            # Nothing to confirm yet (e.g. a brand-new thread): start this section.
            states = _enter(states, current)
            logger.info("router: starting %s", current.value)
            return {**updates, **_switch_to(current, current, states)}

        states = _with_status(states, current, SectionStatus.DONE)
        target = get_next_unfinished_section(states)
        if target is None:
            logger.info("router: all sections done")
            return {
                **updates,
                "section_states": states,
                "context_packet": None,
                "router_directive": RouterDirective.STAY.value,
                "finished": True,
            }
        logger.info("router: %s done, advancing to %s", current.value, target.value)
        return {**updates, **_switch_to(target, current, _enter(states, target), msgs)}

    # --- modify:<section>: reopen an earlier section ------------------------------
    if directive.startswith(MODIFY_PREFIX):
        raw_target = directive[len(MODIFY_PREFIX):]
        try:
            target = SectionID(raw_target)
        except ValueError:
            target = None
        target_state = states.get(target.value) if target else None

        if target is None or target_state is None or target_state.status == SectionStatus.PENDING:
            reason = "unknown section" if target is None else "section not started yet"
            logger.warning("router: rejected modify:%s (%s)", raw_target, reason)
            updates.update(
                router_directive=RouterDirective.STAY.value,
                last_error=f"Cannot modify {raw_target!r}: {reason}.",
            )
            if state.get("context_packet") is None:
                updates["context_packet"] = build_context_packet(current, states)
            return updates

        states = _with_status(states, target, SectionStatus.IN_PROGRESS)
        logger.info("router: reopening %s from %s", target.value, current.value)
        return {**updates, **_switch_to(target, current, states, msgs)}

    # --- stay (default) ---------------------------------------------------------------
    # Always rebuild the packet (cheap, no I/O) so it reflects the latest saved draft
    # and confirmed summaries, e.g. for a returning user whose draft changed since.
    entered = _enter(states, current)
    if entered is not states:
        updates["section_states"] = entered
    updates["context_packet"] = build_context_packet(current, entered)
    updates["router_directive"] = RouterDirective.STAY.value
    return updates

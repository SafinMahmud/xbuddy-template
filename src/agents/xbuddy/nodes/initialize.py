"""Initialize node — validates and sets up conversation state.

This node runs once at the start of every invocation.
"""

import logging
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from ..enums import SectionID
from ..models import JobBuddyData, SectionState, XBuddyState, default_state

logger = logging.getLogger(__name__)


async def initialize_node(
    state: XBuddyState, config: RunnableConfig
) -> dict:
    """Initialize and validate conversation state."""

    is_new_thread = "section_states" not in state
    configurable = config.get("configurable", {})

    thread_id = (
        state.get("thread_id")
        or configurable.get("thread_id")
        or str(uuid4())
    )

    user_id = (
        state.get("user_id")
        or configurable.get("user_id")
        or 1
    )

    updates = {
        "thread_id": thread_id,
        "user_id": user_id,
    }

    for key, default in default_state().items():
        updates[key] = state.get(key, default)

    # Validate current_section.
    value = updates.get("current_section")

    if value is not None:
        try:
            updates["current_section"] = SectionID(value)
        except (ValueError, TypeError):
            expected = ", ".join(section.value for section in SectionID)
            raise ValueError(
                f"Invalid current_section {value!r}. "
                f"Expected one of: {expected}"
            ) from None

    # Validate and restore section states.
    section_states = updates.get("section_states", {})
    validated_section_states = {}

    for key, section_state in section_states.items():
        try:
            section_id = SectionID(key)
        except (ValueError, TypeError):
            expected = ", ".join(section.value for section in SectionID)
            raise ValueError(
                f"Invalid section_states key {key!r}. "
                f"Expected one of: {expected}"
            ) from None

        if isinstance(section_state, dict):
            section_state = SectionState.model_validate(section_state)

        if section_state.section_id != section_id:
            raise ValueError(
                f"Section state key {key!r} does not match "
                f"section_id {section_state.section_id!r}"
            )

        validated_section_states[section_id.value] = section_state

    # Add missing sections as pending.
    for section_id in SectionID:
        if section_id.value not in validated_section_states:
            validated_section_states[section_id.value] = SectionState(
                section_id=section_id
            )

    updates["section_states"] = validated_section_states

    # Restore user_data if serialized as a dict.
    user_data = updates.get("user_data")

    if isinstance(user_data, dict):
        updates["user_data"] = JobBuddyData.model_validate(user_data)

    done_count = sum(
        1
        for section_state in validated_section_states.values()
        if section_state.status == "done"
    )

    logger.info(
        "Thread %s %s with %d sections done",
        thread_id,
        "initialized" if is_new_thread else "resumed",
        done_count,
    )

    return updates
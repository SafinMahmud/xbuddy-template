"""Section templates and navigation helpers."""

from .enums import SectionID, SectionStatus
from .sections import (
    APPLICATION_STRATEGY_TEMPLATE,
    BACKGROUND_TEMPLATE,
    INTERVIEW_PREP_TEMPLATE,
    SKILL_GAP_TEMPLATE,
    TARGET_ROLE_TEMPLATE,
)
from .sections.base_prompt import BASE_RULES, SectionTemplate

SECTION_TEMPLATES: dict[SectionID, SectionTemplate] = {
    SectionID.BACKGROUND: BACKGROUND_TEMPLATE,
    SectionID.TARGET_ROLE: TARGET_ROLE_TEMPLATE,
    SectionID.SKILL_GAP: SKILL_GAP_TEMPLATE,
    SectionID.APPLICATION_STRATEGY: APPLICATION_STRATEGY_TEMPLATE,
    SectionID.INTERVIEW_PREP: INTERVIEW_PREP_TEMPLATE,
}

__all__ = [
    "BASE_RULES",
    "SECTION_TEMPLATES",
    "get_next_section",
    "get_next_unfinished_section",
    "get_section_template",
]


def get_section_template(section_id: SectionID | str) -> SectionTemplate:
    """Return the template for a section. Raises ValueError for unknown ids."""
    try:
        return SECTION_TEMPLATES[SectionID(section_id)]
    except ValueError:
        valid = ", ".join(s.value for s in SectionID)
        raise ValueError(f"Unknown section {section_id!r}. Expected one of: {valid}") from None


def get_next_section(current: SectionID) -> SectionID | None:
    """Return the next section in sequence, or None after the last one."""
    order = list(SectionID)
    idx = order.index(current)
    return order[idx + 1] if idx + 1 < len(order) else None


def get_next_unfinished_section(
    section_states: dict, exclude: SectionID | None = None
) -> SectionID | None:
    """First section in order that isn't done, skipping `exclude`.

    Used after a corrected answer: once the reopened section is confirmed,
    the user returns to where they left off instead of re-walking done sections.
    """
    for section_id in SectionID:
        if section_id == exclude:
            continue
        state = section_states.get(section_id.value)
        if state is None or state.status != SectionStatus.DONE:
            return section_id
    return None

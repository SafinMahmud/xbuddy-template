"""Enumerations for your XBuddy Agent."""

from enum import StrEnum


class SectionStatus(StrEnum):
    """Status of an agent section."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class RouterDirective(StrEnum):
    """Router directive for navigation control."""
    STAY = "stay"
    NEXT = "next"
    MODIFY = "modify"  # Format: "modify:section_id"


class SectionID(StrEnum):
    BACKGROUND = "background"
    TARGET_ROLE = "target_role"
    SKILL_GAP = "skill_gap"
    APPLICATION_STRATEGY = "application_strategy"
    INTERVIEW_PREP = "interview_prep"


class RequirementPriority(StrEnum):
    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"


class WorkType(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    FLEXIBLE = "flexible"

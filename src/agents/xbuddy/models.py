"""Pydantic models for your XBuddy Agent.

Study FounderBuddy's models.py to understand how these work:
https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/models.py
"""

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from .enums import (
    RequirementPriority,
    RouterDirective,
    SectionID,
    SectionStatus,
    WorkType,
)


class SectionContent(BaseModel):
    """Content for an agent section."""
    content: dict[str, Any]  # Rich text content (Tiptap JSON format)
    plain_text: str | None = None  # Plain text version for LLM processing


class SectionState(BaseModel):
    """State of a single section."""
    section_id: SectionID
    content: SectionContent | None = None
    satisfaction_status: str | None = None  # satisfied, needs_improvement, or None
    status: SectionStatus = SectionStatus.PENDING
    confirmed_summary: str | None = None
    # Structured fields the decision reported as covered but extraction could not find.
    # covered_fields is the model's claim; extraction is the check on the actual data.
    unverified_fields: list[str] = Field(default_factory=list)


class ContextPacket(BaseModel):
    """Context packet loaded by the router for the current section."""
    section_id: SectionID
    status: SectionStatus
    system_prompt: str
    draft: SectionContent | None = None
    validation_rules: dict[str, Any] | None = None


MAX_EVIDENCE_QUOTE_CHARS = 300

class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")   # unknown fields (e.g. salary) fail loudly

class UserProfile(_StrictModel):
    current_role: str | None = None
    role_history: list[str] = Field(default_factory=list)
    years_experience: int | None = Field(None, ge=0, le=60)
    skills: list[str] = Field(default_factory=list)
    education: str | None = None

class TargetRole(_StrictModel):
    titles: list[str] = Field(default_factory=list)
    location_region: str | None = None          # coarse only
    work_type: WorkType | None = None
    priorities: list[str] = Field(default_factory=list)

class JobSource(_StrictModel):
    label: str = Field(..., min_length=1)       # "Shopify - Backend Developer"
    url: str | None = None

class RequirementEvidence(_StrictModel):
    source_label: str = Field(..., min_length=1)
    quote: str = Field(..., min_length=1, max_length=MAX_EVIDENCE_QUOTE_CHARS)

class JobRequirement(_StrictModel):
    name: str = Field(..., min_length=1)
    priority: RequirementPriority
    evidence: list[RequirementEvidence] = Field(..., min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _drop_derived_frequency(cls, data):
        if isinstance(data, dict) and "frequency" in data:
            data = {k: v for k, v in data.items() if k != "frequency"}
        return data

    @computed_field
    @property
    def frequency(self) -> int:
        return len({e.source_label for e in self.evidence})

class JobBuddyData(_StrictModel):
    profile: UserProfile = Field(default_factory=UserProfile)
    target_role: TargetRole = Field(default_factory=TargetRole)
    job_sources: list[JobSource] = Field(default_factory=list)
    requirements: list[JobRequirement] = Field(default_factory=list)


class ChatAgentDecision(BaseModel):
    """Structured decision from the generate_decision node."""
    router_directive: str = Field(
        ...,
        description="Navigation control: 'stay', 'next', or 'modify:<section_id>'",
    )
    user_satisfaction_feedback: str | None = Field(
        None, description="User's feedback about satisfaction with the section."
    )
    is_satisfied: bool | None = Field(
        None, description="Whether the user is satisfied with the current section."
    )
    should_save_content: bool = Field(
        False,
        description="Whether to save the current section content.",
    )
    section_summary: str | None = Field(
        None,
        description=(
            "Short plain-text summary of everything collected so far in the current "
            "section. Saved as the draft, and as the confirmed answer on 'next'."
        ),
    )
    covered_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Names of the section's completion-checklist items that have been collected. "
            "'next' is refused in code unless every required item is listed."
        ),
    )

    @field_validator("router_directive")
    def validate_router_directive(cls, v):
        if v not in ["stay", "next"] and not v.startswith("modify:"):
            raise ValueError("router_directive must be 'stay', 'next', or 'modify:<section_id>'")
        return v


class ChatAgentOutput(BaseModel):
    """Complete output from the generate_reply + generate_decision nodes."""
    reply: str = Field(..., description="Conversational response to the user.")
    router_directive: str = Field(
        ...,
        description="Navigation control: 'stay', 'next', or 'modify:<section_id>'",
    )
    user_satisfaction_feedback: str | None = None
    is_satisfied: bool | None = None
    should_save_content: bool = False
    section_summary: str | None = None
    covered_fields: list[str] = Field(default_factory=list)

    @field_validator("router_directive")
    def validate_router_directive(cls, v):
        if v not in ["stay", "next"] and not v.startswith("modify:"):
            raise ValueError("router_directive must be 'stay', 'next', or 'modify:<section_id>'")
        return v


class XBuddyState(MessagesState, total=False):
    user_id: int
    thread_id: str
    current_section: SectionID
    context_packet: ContextPacket | None
    section_states: dict[str, SectionState]
    router_directive: str
    finished: bool
    user_data: JobBuddyData
    short_memory: list[BaseMessage]
    agent_output: ChatAgentOutput | None
    awaiting_user_input: bool
    awaiting_satisfaction_feedback: bool
    error_count: int
    last_error: str | None
    roadmap: str | None                 # renamed from final_output
    should_generate_final_output: bool

def default_state() -> dict[str, Any]:
    """Fresh defaults for a brand-new thread (identity is set by initialize_node)."""
    return {
        "current_section": SectionID.BACKGROUND,
        "context_packet": None,
        "section_states": {s.value: SectionState(section_id=s) for s in SectionID},
        "router_directive": RouterDirective.NEXT.value,
        "finished": False,
        "user_data": JobBuddyData(),
        "short_memory": [],
        "agent_output": None,
        "awaiting_user_input": False,
        "awaiting_satisfaction_feedback": False,
        "error_count": 0,
        "last_error": None,
        "roadmap": None,
        "should_generate_final_output": False,
    }



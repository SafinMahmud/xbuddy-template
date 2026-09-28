"""PR 1 tests: JobBuddy state schema and initialize_node."""

import pytest
from pydantic import ValidationError

from agents.xbuddy.enums import RequirementPriority, SectionID, SectionStatus
from agents.xbuddy.models import (
    JobBuddyData,
    JobRequirement,
    RequirementEvidence,
    SectionState,
    UserProfile,
)
from agents.xbuddy.nodes.initialize import initialize_node

CONFIG = {"configurable": {"thread_id": "thread-abc", "user_id": 42}}


def _returning_state() -> dict:
    """A user who finished sections 1-2 and is halfway through section 3."""
    states = {s.value: SectionState(section_id=s) for s in SectionID}
    for sid in (SectionID.BACKGROUND, SectionID.TARGET_ROLE):
        states[sid.value] = SectionState(
            section_id=sid, status=SectionStatus.DONE, confirmed_summary=f"{sid.value} ok"
        )
    states[SectionID.SKILL_GAP.value] = SectionState(
        section_id=SectionID.SKILL_GAP, status=SectionStatus.IN_PROGRESS
    )
    return {
        "messages": [],
        "thread_id": "thread-abc",
        "user_id": 42,
        "current_section": SectionID.SKILL_GAP,
        "section_states": states,
        "router_directive": "stay",
        "user_data": JobBuddyData(profile=UserProfile(skills=["Python", "Django"])),
    }


# 1. New user gets clean defaults
@pytest.mark.asyncio
async def test_new_user_gets_clean_defaults():
    result = await initialize_node({"messages": []}, CONFIG)

    assert result["thread_id"] == "thread-abc"
    assert result["user_id"] == 42
    assert result["current_section"] == SectionID.BACKGROUND
    assert set(result["section_states"]) == {s.value for s in SectionID}
    assert all(s.status == SectionStatus.PENDING for s in result["section_states"].values())
    assert result["user_data"] == JobBuddyData()
    assert result["finished"] is False
    assert result["roadmap"] is None
    assert "messages" not in result  # never rewrite chat history


@pytest.mark.asyncio
async def test_new_user_without_config_gets_generated_thread_id():
    result = await initialize_node({"messages": []}, {})
    assert result["thread_id"]
    assert result["user_id"] == 1


# 2. Returning user keeps existing progress
@pytest.mark.asyncio
async def test_returning_user_keeps_progress():
    result = await initialize_node(_returning_state(), CONFIG)

    assert result["current_section"] == SectionID.SKILL_GAP
    assert result["router_directive"] == "stay"
    assert result["section_states"]["skill_gap"].status == SectionStatus.IN_PROGRESS
    assert result["user_data"].profile.skills == ["Python", "Django"]


@pytest.mark.asyncio
async def test_saved_identity_wins_over_config():
    state = _returning_state()
    result = await initialize_node(state, {"configurable": {"thread_id": "other", "user_id": 7}})
    assert result["thread_id"] == "thread-abc"
    assert result["user_id"] == 42


# 3. Re-initialization does not erase completed sections
@pytest.mark.asyncio
async def test_reinitialization_is_idempotent():
    first = await initialize_node(_returning_state(), CONFIG)
    second = await initialize_node({"messages": [], **first}, CONFIG)

    for sid in ("background", "target_role"):
        assert second["section_states"][sid].status == SectionStatus.DONE
        assert second["section_states"][sid].confirmed_summary == f"{sid} ok"
    assert second == first


@pytest.mark.asyncio
async def test_missing_sections_are_added_without_touching_saved_ones():
    state = _returning_state()
    del state["section_states"]["interview_prep"]
    result = await initialize_node(state, CONFIG)

    assert result["section_states"]["interview_prep"].status == SectionStatus.PENDING
    assert result["section_states"]["background"].status == SectionStatus.DONE


@pytest.mark.asyncio
async def test_state_restored_as_plain_dicts_is_normalized():
    """Checkpointers may hand back dicts instead of model instances."""
    state = _returning_state()
    state["section_states"] = {k: v.model_dump() for k, v in state["section_states"].items()}
    state["user_data"] = state["user_data"].model_dump()
    result = await initialize_node(state, CONFIG)

    assert isinstance(result["section_states"]["background"], SectionState)
    assert result["section_states"]["background"].status == SectionStatus.DONE
    assert isinstance(result["user_data"], JobBuddyData)


# 4. A requirement retains its source evidence
def test_requirement_keeps_evidence_and_derives_frequency():
    req = JobRequirement(
        name="Kubernetes",
        priority=RequirementPriority.MUST_HAVE,
        evidence=[
            RequirementEvidence(source_label="Shopify - Backend", quote="Experience with Kubernetes"),
            RequirementEvidence(source_label="Wealthsimple - SWE", quote="K8s in production"),
            RequirementEvidence(source_label="Shopify - Backend", quote="deploy to Kubernetes"),
        ],
    )
    assert req.frequency == 2  # distinct postings, not raw mentions
    dumped = req.model_dump()
    assert dumped["frequency"] == 2
    assert dumped["evidence"][0]["source_label"] == "Shopify - Backend"


@pytest.mark.asyncio
async def test_requirement_evidence_survives_initialize():
    state = _returning_state()
    state["user_data"] = JobBuddyData(
        requirements=[
            JobRequirement(
                name="Docker",
                priority=RequirementPriority.NICE_TO_HAVE,
                evidence=[RequirementEvidence(source_label="Acme - SWE", quote="Docker a plus")],
            )
        ]
    ).model_dump()
    result = await initialize_node(state, CONFIG)
    assert result["user_data"].requirements[0].evidence[0].source_label == "Acme - SWE"


def test_requirement_without_evidence_is_rejected():
    with pytest.raises(ValidationError):
        JobRequirement(name="Go", priority=RequirementPriority.MUST_HAVE, evidence=[])


def test_evidence_quote_is_capped_so_raw_postings_are_not_stored():
    with pytest.raises(ValidationError):
        RequirementEvidence(source_label="Acme", quote="x" * 301)


def test_unexpected_personal_fields_are_rejected():
    with pytest.raises(ValidationError):
        UserProfile(salary_expectation="120k")


# 5. Invalid section or status values fail clearly
@pytest.mark.asyncio
async def test_invalid_current_section_fails_clearly():
    state = _returning_state()
    state["current_section"] = "section_1"
    with pytest.raises(ValueError, match="Invalid current_section 'section_1'.*background"):
        await initialize_node(state, CONFIG)


@pytest.mark.asyncio
async def test_invalid_section_state_key_fails_clearly():
    state = _returning_state()
    state["section_states"]["bogus"] = {"section_id": "background"}
    with pytest.raises(ValueError, match="Invalid section_states key 'bogus'"):
        await initialize_node(state, CONFIG)


@pytest.mark.asyncio
async def test_invalid_status_fails_clearly():
    state = _returning_state()
    state["section_states"]["background"] = {"section_id": "background", "status": "finished"}
    with pytest.raises(ValidationError, match="status"):
        await initialize_node(state, CONFIG)


def test_graph_still_compiles():
    from agents.xbuddy.agent import graph

    assert "initialize" in graph.nodes
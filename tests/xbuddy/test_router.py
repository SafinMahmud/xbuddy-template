"""PR 2 tests: router_node directives (stay, advance, corrected answer)."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import default_state
from agents.xbuddy.nodes.router import router_node
from agents.xbuddy.tools import build_context_packet

CONFIG = {"configurable": {"thread_id": "t1", "user_id": 1}}


def make_state(current: SectionID, statuses: dict[SectionID, SectionStatus], **extra) -> dict:
    """State with given section statuses; done sections get a confirmed summary."""
    state = {"messages": [], "thread_id": "t1", "user_id": 1, **default_state()}
    for sid, status in statuses.items():
        summary = f"{sid.value} summary" if status == SectionStatus.DONE else None
        state["section_states"][sid.value] = state["section_states"][sid.value].model_copy(
            update={"status": status, "confirmed_summary": summary}
        )
    state["current_section"] = current
    state.update(extra)
    return state


def status(result: dict, sid: SectionID) -> SectionStatus:
    return result["section_states"][sid.value].status


# --- stay ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stay_keeps_section_and_loads_missing_context():
    state = make_state(
        SectionID.TARGET_ROLE,
        {SectionID.BACKGROUND: SectionStatus.DONE, SectionID.TARGET_ROLE: SectionStatus.IN_PROGRESS},
        router_directive="stay",
        messages=[AIMessage("What titles?"), HumanMessage("Backend developer")],
    )
    result = await router_node(state, CONFIG)

    assert "current_section" not in result  # unchanged
    assert result["context_packet"].section_id == SectionID.TARGET_ROLE
    assert result["router_directive"] == "stay"
    assert result["awaiting_user_input"] is False


@pytest.mark.asyncio
async def test_stay_does_not_reload_existing_context():
    state = make_state(SectionID.BACKGROUND, {SectionID.BACKGROUND: SectionStatus.IN_PROGRESS})
    state["context_packet"] = build_context_packet(SectionID.BACKGROUND, state["section_states"])
    state["router_directive"] = "stay"
    result = await router_node(state, CONFIG)

    assert "context_packet" not in result
    assert "section_states" not in result


@pytest.mark.asyncio
async def test_stay_reloads_context_for_a_different_section():
    state = make_state(SectionID.TARGET_ROLE, {SectionID.BACKGROUND: SectionStatus.DONE})
    state["context_packet"] = build_context_packet(SectionID.BACKGROUND, state["section_states"])
    result = await router_node({**state, "router_directive": "stay"}, CONFIG)

    assert result["context_packet"].section_id == SectionID.TARGET_ROLE
    assert status(result, SectionID.TARGET_ROLE) == SectionStatus.IN_PROGRESS


# --- advance --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_next_on_new_thread_starts_first_section_without_skipping():
    state = make_state(SectionID.BACKGROUND, {}, router_directive="next")
    result = await router_node(state, CONFIG)

    assert result["current_section"] == SectionID.BACKGROUND
    assert status(result, SectionID.BACKGROUND) == SectionStatus.IN_PROGRESS
    assert result["router_directive"] == "stay"


@pytest.mark.asyncio
async def test_next_marks_done_and_advances():
    state = make_state(
        SectionID.BACKGROUND,
        {SectionID.BACKGROUND: SectionStatus.IN_PROGRESS},
        router_directive="next",
        short_memory=[HumanMessage("old section chatter")],
    )
    result = await router_node(state, CONFIG)

    assert status(result, SectionID.BACKGROUND) == SectionStatus.DONE
    assert result["current_section"] == SectionID.TARGET_ROLE
    assert status(result, SectionID.TARGET_ROLE) == SectionStatus.IN_PROGRESS
    assert result["context_packet"].section_id == SectionID.TARGET_ROLE
    assert result["short_memory"] == []
    assert result["router_directive"] == "stay"


@pytest.mark.asyncio
async def test_next_from_last_section_finishes():
    statuses = {sid: SectionStatus.DONE for sid in SectionID}
    statuses[SectionID.INTERVIEW_PREP] = SectionStatus.IN_PROGRESS
    state = make_state(SectionID.INTERVIEW_PREP, statuses, router_directive="next")
    result = await router_node(state, CONFIG)

    assert result["finished"] is True
    assert all(s.status == SectionStatus.DONE for s in result["section_states"].values())
    assert result["context_packet"] is None


# --- corrected answer (modify) --------------------------------------------------------

@pytest.mark.asyncio
async def test_modify_reopens_earlier_section_with_previous_answer():
    state = make_state(
        SectionID.SKILL_GAP,
        {
            SectionID.BACKGROUND: SectionStatus.DONE,
            SectionID.TARGET_ROLE: SectionStatus.DONE,
            SectionID.SKILL_GAP: SectionStatus.IN_PROGRESS,
        },
        router_directive="modify:target_role",
    )
    result = await router_node(state, CONFIG)

    assert result["current_section"] == SectionID.TARGET_ROLE
    assert status(result, SectionID.TARGET_ROLE) == SectionStatus.IN_PROGRESS
    assert status(result, SectionID.SKILL_GAP) == SectionStatus.IN_PROGRESS  # untouched
    packet = result["context_packet"]
    assert packet.draft.plain_text == "target_role summary"
    assert "update it, do not start over" in packet.system_prompt
    assert result["router_directive"] == "stay"


@pytest.mark.asyncio
async def test_after_correction_next_returns_to_where_user_left_off():
    """Correct target_role while in skill_gap, confirm, and land back in skill_gap."""
    state = make_state(
        SectionID.SKILL_GAP,
        {
            SectionID.BACKGROUND: SectionStatus.DONE,
            SectionID.TARGET_ROLE: SectionStatus.DONE,
            SectionID.SKILL_GAP: SectionStatus.IN_PROGRESS,
        },
        router_directive="modify:target_role",
    )
    reopened = await router_node(state, CONFIG)
    confirmed = await router_node({**state, **reopened, "router_directive": "next"}, CONFIG)

    assert confirmed["current_section"] == SectionID.SKILL_GAP
    assert status(confirmed, SectionID.TARGET_ROLE) == SectionStatus.DONE
    assert status(confirmed, SectionID.APPLICATION_STRATEGY) == SectionStatus.PENDING


@pytest.mark.asyncio
async def test_modify_unknown_section_stays_and_records_error():
    state = make_state(SectionID.BACKGROUND, {SectionID.BACKGROUND: SectionStatus.IN_PROGRESS},
                       router_directive="modify:salary")
    result = await router_node(state, CONFIG)

    assert "current_section" not in result
    assert result["router_directive"] == "stay"
    assert "salary" in result["last_error"]
    assert result["context_packet"].section_id == SectionID.BACKGROUND


@pytest.mark.asyncio
async def test_modify_cannot_jump_ahead_to_unstarted_section():
    state = make_state(SectionID.BACKGROUND, {SectionID.BACKGROUND: SectionStatus.IN_PROGRESS},
                       router_directive="modify:interview_prep")
    result = await router_node(state, CONFIG)

    assert "current_section" not in result
    assert "not started" in result["last_error"]


@pytest.mark.asyncio
async def test_router_does_not_mutate_input_state():
    state = make_state(SectionID.BACKGROUND, {SectionID.BACKGROUND: SectionStatus.IN_PROGRESS},
                       router_directive="next")
    await router_node(state, CONFIG)

    assert state["section_states"]["background"].status == SectionStatus.IN_PROGRESS
    assert state["current_section"] == SectionID.BACKGROUND


# --- graph wiring ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_new_thread_runs_initialize_then_router():
    from agents.xbuddy.agent import graph

    cfg = {"configurable": {"thread_id": "graph-new", "user_id": 1}}
    result = await graph.ainvoke({"messages": []}, cfg)

    assert result["current_section"] == SectionID.BACKGROUND
    assert result["section_states"]["background"].status == SectionStatus.IN_PROGRESS
    assert result["context_packet"].section_id == SectionID.BACKGROUND

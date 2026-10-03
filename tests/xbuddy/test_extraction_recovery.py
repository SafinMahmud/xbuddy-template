"""PR 4 tests: recovering structured data after a failed or incomplete extraction."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import ChatAgentOutput, SectionState, default_state
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.nodes.memory_updater import MAX_EXTRACTION_ATTEMPTS, memory_updater_node
from agents.xbuddy.nodes.router import router_node
from agents.xbuddy.tools import build_context_packet

CONFIG = {"configurable": {"thread_id": "t", "user_id": 1}}
BG_STRUCTURED = ["current_role", "role_history", "years_experience", "skills", "education"]
PROFILE = {"current_role": "Backend Developer", "role_history": ["Junior dev"],
           "years_experience": 3, "skills": ["Python"], "education": "MSc CS"}


class CountingModel(FakeListChatModel):
    """Fails while `fail` is set, and counts every call."""

    calls: list = Field(default_factory=list)
    fail: bool = False

    async def ainvoke(self, input, config=None, **kwargs):
        self.calls.append(input)
        if self.fail:
            raise TimeoutError("provider timed out")
        return await super().ainvoke(input, config, **kwargs)


@pytest.fixture(autouse=True)
def no_supabase(monkeypatch):
    monkeypatch.setattr(mu, "get_section_store", lambda: None)


def use_extractor(monkeypatch, payload=PROFILE, fail=False) -> CountingModel:
    model = CountingModel(responses=[json.dumps(payload)], fail=fail)
    monkeypatch.setattr(extraction_mod, "get_chat_model", lambda config=None: model)
    return model


def state_with(current: SectionID, **output) -> dict:
    state = {"messages": [HumanMessage("Backend dev, 3 years, Python, MSc CS, was a junior dev")],
             "thread_id": "t", "user_id": 1, **default_state()}
    state["current_section"] = current
    state["section_states"][current.value] = SectionState(
        section_id=current, status=SectionStatus.IN_PROGRESS)
    state["short_memory"] = [AIMessage("Summary. Right?"), HumanMessage("Yes")]
    state["agent_output"] = ChatAgentOutput(reply="ok", **{"router_directive": "stay", **output})
    return state


def confirm(state: dict, summary="Backend dev, 3 years") -> dict:
    state["agent_output"] = ChatAgentOutput(
        reply="ok", router_directive="next", is_satisfied=True, section_summary=summary)
    return state


def stay(state: dict) -> dict:
    state["agent_output"] = ChatAgentOutput(reply="ok", router_directive="stay")
    return state


async def failed_confirmation(monkeypatch) -> dict:
    """Background confirmed while the provider is down, then the user moves on."""
    use_extractor(monkeypatch, fail=True)
    state = confirm(state_with(SectionID.BACKGROUND))
    state.update(await memory_updater_node(state, CONFIG))
    state["current_section"] = SectionID.TARGET_ROLE
    state["section_states"]["target_role"] = SectionState(
        section_id=SectionID.TARGET_ROLE, status=SectionStatus.IN_PROGRESS)
    return state


# --- automatic recovery ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_extraction_is_retried_and_recovered_on_the_next_turn(monkeypatch):
    state = await failed_confirmation(monkeypatch)
    background = state["section_states"]["background"]
    assert background.status == SectionStatus.DONE  # the confirmation is kept
    assert background.unverified_fields == BG_STRUCTURED
    assert background.extraction_attempts == 1
    assert state["user_data"].profile.current_role is None

    use_extractor(monkeypatch)  # provider is back
    result = await memory_updater_node(stay(state), CONFIG)

    recovered = result["section_states"]["background"]
    assert recovered.unverified_fields == []
    assert recovered.extraction_attempts == 2
    assert recovered.status == SectionStatus.DONE
    assert recovered.confirmed_summary == "Backend dev, 3 years"
    assert result["user_data"].profile.current_role == "Backend Developer"
    assert result["user_data"].profile.education == "MSc CS"


@pytest.mark.asyncio
async def test_retry_uses_the_confirmed_summary_and_full_conversation(monkeypatch):
    state = await failed_confirmation(monkeypatch)
    model = use_extractor(monkeypatch)
    await memory_updater_node(stay(state), CONFIG)

    prompt_text = model.calls[0][1].content
    assert "Backend dev, 3 years" in prompt_text  # confirmed summary
    assert "was a junior dev" in prompt_text  # from messages; short memory has moved on


@pytest.mark.asyncio
async def test_retries_are_bounded(monkeypatch):
    state = await failed_confirmation(monkeypatch)
    model = use_extractor(monkeypatch, fail=True)  # provider stays down

    for _ in range(5):
        state.update(await memory_updater_node(stay(state), CONFIG))

    background = state["section_states"]["background"]
    assert background.extraction_attempts == MAX_EXTRACTION_ATTEMPTS
    assert len(model.calls) == MAX_EXTRACTION_ATTEMPTS - 1  # retries only, then it stops
    assert background.unverified_fields == BG_STRUCTURED
    assert background.status == SectionStatus.DONE


@pytest.mark.asyncio
async def test_no_retry_when_nothing_is_unverified(monkeypatch):
    use_extractor(monkeypatch)
    state = confirm(state_with(SectionID.BACKGROUND))
    state.update(await memory_updater_node(state, CONFIG))
    model = use_extractor(monkeypatch)

    result = await memory_updater_node(stay(state), CONFIG)
    assert model.calls == []
    assert result == {}


# --- repair by the user ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_can_repair_by_correcting_the_section(monkeypatch):
    """After retries are exhausted, reopening and re-confirming re-runs extraction."""
    state = await failed_confirmation(monkeypatch)
    use_extractor(monkeypatch, fail=True)
    for _ in range(3):
        state.update(await memory_updater_node(stay(state), CONFIG))
    assert state["section_states"]["background"].extraction_attempts == MAX_EXTRACTION_ATTEMPTS

    # The user says "I want to fix my background": modify reopens it.
    state.update(await router_node({**state, "router_directive": "modify:background"}, CONFIG))
    assert state["current_section"] == SectionID.BACKGROUND

    use_extractor(monkeypatch)
    result = await memory_updater_node(confirm(state, "Backend dev, 3 years, MSc CS"), CONFIG)

    repaired = result["section_states"]["background"]
    assert repaired.unverified_fields == []
    assert repaired.extraction_attempts == 1  # a new confirmation resets the budget
    assert repaired.confirmed_summary == "Backend dev, 3 years, MSc CS"
    assert result["user_data"].profile.education == "MSc CS"


def test_unsaved_details_are_shown_to_the_reply_model():
    states = default_state()["section_states"]
    states["background"] = SectionState(
        section_id=SectionID.BACKGROUND, status=SectionStatus.DONE,
        confirmed_summary="Backend dev", unverified_fields=["education", "skills"])
    prompt = build_context_packet(SectionID.TARGET_ROLE, states).system_prompt

    assert "DETAILS NOT SAVED YET" in prompt
    assert "- Background: education, skills" in prompt
    clean = build_context_packet(SectionID.TARGET_ROLE, default_state()["section_states"])
    assert "DETAILS NOT SAVED YET" not in clean.system_prompt

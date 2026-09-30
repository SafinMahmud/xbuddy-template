"""PR 4 tests: memory_updater, structured extraction, and Supabase persistence."""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy import persistence
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.extraction import merge_extraction
from agents.xbuddy.models import ChatAgentOutput, JobBuddyData, SectionState, default_state
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.nodes.memory_updater import memory_updater_node, to_tiptap

CONFIG = {"configurable": {"thread_id": "t", "user_id": 1}}


class FakeStore:
    def __init__(self, fail: bool = False):
        self.saved: list[SectionState] = []
        self.fail = fail

    async def save_section(self, user_id, thread_id, state):
        if self.fail:
            raise RuntimeError("supabase down")
        self.saved.append(state)


class BrokenModel(FakeListChatModel):
    async def ainvoke(self, input, config=None, **kwargs):
        raise TimeoutError("provider timed out")


@pytest.fixture
def store(monkeypatch):
    s = FakeStore()
    monkeypatch.setattr(mu, "get_section_store", lambda: s)
    return s


def extraction_returns(monkeypatch, payload):
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(
        extraction_mod, "get_chat_model", lambda config=None: FakeListChatModel(responses=[raw])
    )


def make_state(current=SectionID.BACKGROUND, done=(), **output) -> dict:
    state = {"messages": [], "thread_id": "t", "user_id": 1, **default_state()}
    for sid in SectionID:
        status = SectionStatus.DONE if sid in done else SectionStatus.PENDING
        if sid == current:
            status = SectionStatus.IN_PROGRESS
        state["section_states"][sid.value] = SectionState(section_id=sid, status=status)
    state["current_section"] = current
    state["short_memory"] = [AIMessage("Your title?"), HumanMessage("Backend dev, 3 years")]
    state["agent_output"] = ChatAgentOutput(reply="ok", **{"router_directive": "stay", **output})
    return state


# --- drafts --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stay_saves_unconfirmed_draft(store):
    state = make_state(should_save_content=True, section_summary="Backend dev\n3 years")
    result = await memory_updater_node(state, CONFIG)

    section = result["section_states"]["background"]
    assert section.content.plain_text == "Backend dev\n3 years"
    assert section.content.content == to_tiptap("Backend dev\n3 years")
    assert section.status == SectionStatus.IN_PROGRESS
    assert section.confirmed_summary is None  # a draft is not a confirmation
    assert store.saved == [section]


@pytest.mark.asyncio
async def test_nothing_to_save_is_a_noop(store):
    result = await memory_updater_node(make_state(), CONFIG)
    assert result == {}
    assert store.saved == []


# --- confirmation + extraction ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_next_confirms_section_and_extracts_profile(monkeypatch, store):
    extraction_returns(monkeypatch, {"current_role": "Backend Developer", "years_experience": 3,
                                     "skills": ["Python", "Django"], "role_history": [],
                                     "education": "MSc Computer Science"})
    state = make_state(router_directive="next", is_satisfied=True,
                       should_save_content=True, section_summary="Backend dev, 3 years")
    result = await memory_updater_node(state, CONFIG)

    section = result["section_states"]["background"]
    assert section.status == SectionStatus.DONE
    assert section.confirmed_summary == "Backend dev, 3 years"
    assert section.satisfaction_status == "satisfied"
    assert result["user_data"].profile.skills == ["Python", "Django"]
    assert "should_generate_final_output" not in result
    assert store.saved[-1].status == SectionStatus.DONE


@pytest.mark.asyncio
async def test_extraction_failure_does_not_block_confirmation(monkeypatch, store):
    monkeypatch.setattr(extraction_mod, "get_chat_model",
                        lambda config=None: BrokenModel(responses=["x"]))
    state = make_state(router_directive="next", is_satisfied=True, section_summary="Backend dev")
    result = await memory_updater_node(state, CONFIG)

    assert result["section_states"]["background"].status == SectionStatus.DONE
    assert "extraction[background]" in result["last_error"]
    assert "user_data" not in result


@pytest.mark.asyncio
async def test_malformed_extraction_is_recorded(monkeypatch, store):
    extraction_returns(monkeypatch, "not json at all")
    state = make_state(router_directive="next", is_satisfied=True, section_summary="Backend dev")
    result = await memory_updater_node(state, CONFIG)
    assert "no JSON" in result["last_error"]


# --- the model's coverage claim vs the extracted data ------------------------------------------

@pytest.mark.asyncio
async def test_claimed_field_without_extracted_data_is_recorded_as_unverified(monkeypatch, store):
    """covered_fields said everything was covered, but extraction found no education."""
    extraction_returns(monkeypatch, {"current_role": "Backend Developer", "years_experience": 3,
                                     "skills": ["Python"], "role_history": ["Junior dev"],
                                     "education": None})
    state = make_state(router_directive="next", is_satisfied=True, section_summary="Backend dev",
                       covered_fields=["current_role", "role_history", "years_experience",
                                       "skills", "education"])
    result = await memory_updater_node(state, CONFIG)

    section = result["section_states"]["background"]
    assert section.unverified_fields == ["education"]
    assert result["user_data"].profile.education is None  # the claim is not stored as data
    assert section.status == SectionStatus.DONE  # the user's confirmation still stands


@pytest.mark.asyncio
async def test_fully_extracted_section_has_no_unverified_fields(monkeypatch, store):
    extraction_returns(monkeypatch, {"current_role": "Backend Developer", "years_experience": 3,
                                     "skills": ["Python"], "role_history": ["Junior dev"],
                                     "education": "MSc CS"})
    state = make_state(router_directive="next", is_satisfied=True, section_summary="Backend dev")
    result = await memory_updater_node(state, CONFIG)
    assert result["section_states"]["background"].unverified_fields == []


@pytest.mark.asyncio
async def test_failed_extraction_marks_every_structured_field_unverified(monkeypatch, store):
    monkeypatch.setattr(extraction_mod, "get_chat_model",
                        lambda config=None: BrokenModel(responses=["x"]))
    state = make_state(router_directive="next", is_satisfied=True, section_summary="Backend dev")
    result = await memory_updater_node(state, CONFIG)
    assert result["section_states"]["background"].unverified_fields == [
        "current_role", "role_history", "years_experience", "skills", "education"]


def test_skill_gap_extraction_keeps_only_traceable_requirements():
    data = {
        "job_sources": [{"label": "Shopify - Backend"}, {"label": "Wealthsimple - SWE"}],
        "requirements": [
            {"name": "Kubernetes", "priority": "must_have", "evidence": [
                {"source_label": "Shopify - Backend", "quote": "Kubernetes in production"},
                {"source_label": "Wealthsimple - SWE", "quote": "K8s" + "x" * 400},
                {"source_label": "Made Up Inc", "quote": "hallucinated"},
            ]},
            {"name": "Rust", "priority": "nice_to_have", "evidence": [
                {"source_label": "Unknown Co", "quote": "Rust"},
            ]},
        ],
    }
    result = merge_extraction(SectionID.SKILL_GAP, data, JobBuddyData())

    assert [r.name for r in result.requirements] == ["Kubernetes"]  # Rust had no valid evidence
    k8s = result.requirements[0]
    assert k8s.frequency == 2
    assert {e.source_label for e in k8s.evidence} == {"Shopify - Backend", "Wealthsimple - SWE"}
    assert all(len(e.quote) <= 300 for e in k8s.evidence)


@pytest.mark.asyncio
async def test_sections_without_structured_fields_skip_extraction(monkeypatch, store):
    monkeypatch.setattr(extraction_mod, "get_chat_model",
                        lambda config=None: BrokenModel(responses=["x"]))  # would fail if called
    state = make_state(SectionID.APPLICATION_STRATEGY,
                       done=(SectionID.BACKGROUND, SectionID.TARGET_ROLE, SectionID.SKILL_GAP),
                       router_directive="next", is_satisfied=True, section_summary="10 apps/week")
    result = await memory_updater_node(state, CONFIG)
    assert "last_error" not in result


# --- completion ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_confirming_last_section_triggers_final_output(store):
    others = tuple(s for s in SectionID if s != SectionID.INTERVIEW_PREP)
    state = make_state(SectionID.INTERVIEW_PREP, done=others, router_directive="next",
                       is_satisfied=True, section_summary="Behavioral + system design")
    result = await memory_updater_node(state, CONFIG)

    assert result["should_generate_final_output"] is True
    assert result["section_states"]["interview_prep"].status == SectionStatus.DONE


@pytest.mark.asyncio
async def test_last_section_with_earlier_gap_does_not_finish(store):
    done = (SectionID.BACKGROUND, SectionID.TARGET_ROLE, SectionID.SKILL_GAP)  # app strategy open
    state = make_state(SectionID.INTERVIEW_PREP, done=done, router_directive="next",
                       is_satisfied=True, section_summary="Behavioral")
    result = await memory_updater_node(state, CONFIG)
    assert "should_generate_final_output" not in result


# --- persistence -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supabase_failure_keeps_state_and_records_error(monkeypatch):
    monkeypatch.setattr(mu, "get_section_store", lambda: FakeStore(fail=True))
    state = make_state(should_save_content=True, section_summary="Backend dev")
    result = await memory_updater_node(state, CONFIG)

    assert result["section_states"]["background"].content.plain_text == "Backend dev"
    assert "persist[background]: supabase down" in result["last_error"]


@pytest.mark.asyncio
async def test_no_supabase_configured_uses_checkpointer_only(monkeypatch):
    monkeypatch.setattr(persistence, "get_section_store", lambda: None)
    monkeypatch.setattr(mu, "get_section_store", lambda: None)
    result = await memory_updater_node(make_state(should_save_content=True,
                                                  section_summary="x"), CONFIG)
    assert "last_error" not in result


def test_get_section_store_is_none_without_credentials(monkeypatch):
    from core.settings import settings

    monkeypatch.setattr(settings, "SUPABASE_URL", None)
    assert persistence.get_section_store() is None


@pytest.mark.asyncio
async def test_supabase_store_upserts_on_unique_key(monkeypatch):
    from integrations.supabase import supabase_client

    table = MagicMock()
    fake_client = MagicMock()
    fake_client.table.return_value = table
    monkeypatch.setattr(supabase_client, "get_supabase_client", lambda: fake_client)

    store = persistence.SupabaseSectionStore()
    section = SectionState(section_id=SectionID.BACKGROUND, status=SectionStatus.DONE,
                           confirmed_summary="Backend dev")
    await store.save_section(1, "t", section)

    row, kwargs = table.upsert.call_args.args[0], table.upsert.call_args.kwargs
    assert kwargs["on_conflict"] == "user_id,thread_id,section_id"
    assert row["agent_id"] == "jobbuddy"
    assert row["status"] == "done"
    assert row["confirmed_summary"] == "Backend dev"

"""PR 4 tests: saved state survives a restart and is isolated by user and thread.

Uses the real async SQLite checkpointer on a file. A "restart" closes the
database connection and builds a brand-new graph object on the same file.
"""

import json
from contextlib import asynccontextmanager

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.graph.builder import build_xbuddy_graph
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.prompts import get_section_template

SUMMARY = "Backend dev, 3 years, Python"
BG_FIELDS = get_section_template(SectionID.BACKGROUND).required_fields


class RecordingStore:
    def __init__(self):
        self.rows = []

    async def save_section(self, user_id, thread_id, state):
        self.rows.append((user_id, thread_id, state.section_id.value, state.status.value))


@pytest.fixture
def store(monkeypatch):
    s = RecordingStore()
    monkeypatch.setattr(mu, "get_section_store", lambda: s)
    return s


@asynccontextmanager
async def app(db_path):
    """A fresh process: new connection, new compiled graph, same database file."""
    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
        graph = build_xbuddy_graph()
        graph.checkpointer = saver
        yield graph


def cfg(thread_id: str, user_id: int = 1) -> dict:
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}


def models(monkeypatch, reply: str, decision: dict, profile: dict | None = None):
    for module, text in (
        (reply_mod, reply),
        (decision_mod, json.dumps(decision)),
        (extraction_mod, json.dumps(profile or {})),
    ):
        monkeypatch.setattr(
            module, "get_chat_model",
            lambda config=None, text=text: FakeListChatModel(responses=[text]))


async def draft_turn(graph, config, monkeypatch, summary=SUMMARY):
    models(monkeypatch, "Summary: ... Does this look right?",
           {"router_directive": "stay", "should_save_content": True, "section_summary": summary})
    return await graph.ainvoke({"messages": [HumanMessage("Backend dev, 3 yrs")]}, config)


async def confirm_turn(graph, config, monkeypatch, summary=SUMMARY):
    models(monkeypatch, "Thanks! What job titles are you aiming for next?",
           {"router_directive": "next", "is_satisfied": True, "should_save_content": True,
            "section_summary": summary, "covered_fields": BG_FIELDS},
           {"current_role": "Backend Developer", "role_history": ["Junior Web Developer"],
            "years_experience": 3, "skills": ["Python"], "education": "MSc CS"})
    return await graph.ainvoke({"messages": [HumanMessage("Yes, looks good")]}, config)


# --- restart ---------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unconfirmed_draft_survives_restart(tmp_path, monkeypatch, store):
    db = tmp_path / "jobbuddy.db"
    async with app(db) as graph:
        await draft_turn(graph, cfg("t-draft"), monkeypatch)

    async with app(db) as restarted:  # new connection, new graph object
        state = (await restarted.aget_state(cfg("t-draft"))).values

    background = state["section_states"]["background"]
    assert background.content.plain_text == SUMMARY
    assert background.confirmed_summary is None  # still a draft after the restart
    assert background.status == SectionStatus.IN_PROGRESS
    assert state["current_section"] == SectionID.BACKGROUND
    assert len(state["messages"]) == 2


@pytest.mark.asyncio
async def test_confirmed_state_survives_restart_and_conversation_resumes(
    tmp_path, monkeypatch, store
):
    db = tmp_path / "jobbuddy.db"
    config = cfg("t-confirmed")
    async with app(db) as graph:
        await draft_turn(graph, config, monkeypatch)
        await confirm_turn(graph, config, monkeypatch)

    async with app(db) as restarted:
        state = (await restarted.aget_state(config)).values
        background = state["section_states"]["background"]
        assert background.status == SectionStatus.DONE
        assert background.confirmed_summary == SUMMARY
        assert state["user_data"].profile.current_role == "Backend Developer"
        assert state["current_section"] == SectionID.TARGET_ROLE

        # The returning user continues in Target Role; nothing is reset or re-asked.
        models(monkeypatch, "Got it. Which city or region?",
               {"router_directive": "stay", "should_save_content": True,
                "section_summary": "Backend roles"})
        resumed = await restarted.ainvoke({"messages": [HumanMessage("Backend roles")]}, config)

    assert resumed["current_section"] == SectionID.TARGET_ROLE
    assert resumed["section_states"]["background"].status == SectionStatus.DONE
    assert resumed["section_states"]["target_role"].content.plain_text == "Backend roles"
    assert "Background: " + SUMMARY in resumed["context_packet"].system_prompt


@pytest.mark.asyncio
async def test_failed_extraction_is_recovered_after_a_restart(tmp_path, monkeypatch, store):
    """Provider down at confirmation, app restarts, next turn recovers the profile."""
    db = tmp_path / "jobbuddy.db"
    config = cfg("t-recover")

    class Down(FakeListChatModel):
        async def ainvoke(self, input, config=None, **kwargs):
            raise TimeoutError("provider timed out")

    async with app(db) as graph:
        await draft_turn(graph, config, monkeypatch)
        models(monkeypatch, "Thanks! What job titles are you aiming for next?",
               {"router_directive": "next", "is_satisfied": True, "should_save_content": True,
                "section_summary": SUMMARY, "covered_fields": BG_FIELDS})
        monkeypatch.setattr(extraction_mod, "get_chat_model",
                            lambda config=None: Down(responses=["x"]))
        failed = await graph.ainvoke({"messages": [HumanMessage("Yes, looks good")]}, config)

    assert failed["section_states"]["background"].status == SectionStatus.DONE
    assert failed["section_states"]["background"].unverified_fields
    assert failed["user_data"].profile.current_role is None

    async with app(db) as restarted:  # provider is back after the restart
        models(monkeypatch, "Got it. Which city or region?",
               {"router_directive": "stay", "should_save_content": True,
                "section_summary": "Backend roles"},
               {"current_role": "Backend Developer", "role_history": ["Junior Web Developer"],
                "years_experience": 3, "skills": ["Python"], "education": "MSc CS"})
        resumed = await restarted.ainvoke({"messages": [HumanMessage("Backend roles")]}, config)

    assert resumed["section_states"]["background"].unverified_fields == []
    assert resumed["user_data"].profile.current_role == "Backend Developer"
    assert resumed["current_section"] == SectionID.TARGET_ROLE


# --- isolation -------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_threads_and_users_do_not_share_state(tmp_path, monkeypatch, store):
    db = tmp_path / "jobbuddy.db"
    alice, alice_second, bob = cfg("alice-1", 1), cfg("alice-2", 1), cfg("bob-1", 2)

    async with app(db) as graph:
        await draft_turn(graph, alice, monkeypatch)
        await confirm_turn(graph, alice, monkeypatch)
        await draft_turn(graph, bob, monkeypatch, summary="Nurse, 10 years")
        await graph.ainvoke({"messages": []}, alice_second)  # same user, new thread

    async with app(db) as restarted:
        a = (await restarted.aget_state(alice)).values
        a2 = (await restarted.aget_state(alice_second)).values
        b = (await restarted.aget_state(bob)).values
        unknown = (await restarted.aget_state(cfg("nobody"))).values

    # Alice's first thread kept her confirmed data.
    assert a["current_section"] == SectionID.TARGET_ROLE
    assert a["user_data"].profile.current_role == "Backend Developer"
    # Bob only has his own draft, and none of Alice's data.
    assert b["user_id"] == 2
    assert b["current_section"] == SectionID.BACKGROUND
    assert b["section_states"]["background"].content.plain_text == "Nurse, 10 years"
    assert b["user_data"].profile.current_role is None
    # Alice's second thread starts clean: same user does not mean same conversation.
    assert a2["section_states"]["background"].content is None
    assert a2["user_data"].profile.current_role is None
    assert len(a2["messages"]) == 0
    # A thread that never existed has no state at all.
    assert unknown == {}


@pytest.mark.asyncio
async def test_supabase_rows_are_keyed_by_user_and_thread(tmp_path, monkeypatch, store):
    async with app(tmp_path / "jobbuddy.db") as graph:
        await draft_turn(graph, cfg("alice-1", 1), monkeypatch)
        await draft_turn(graph, cfg("bob-1", 2), monkeypatch, summary="Nurse, 10 years")

    assert store.rows == [
        (1, "alice-1", "background", "in_progress"),
        (2, "bob-1", "background", "in_progress"),
    ]

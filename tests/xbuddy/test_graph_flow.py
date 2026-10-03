"""End-to-end graph turns with fake models (PR 3 + PR 4 wiring)."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.prompts import get_section_template


def fake(monkeypatch, module, *responses):
    model = FakeListChatModel(responses=list(responses))
    monkeypatch.setattr(module, "get_chat_model", lambda config=None: model)


@pytest.fixture(autouse=True)
def no_supabase(monkeypatch):
    monkeypatch.setattr(mu, "get_section_store", lambda: None)


@pytest.mark.asyncio
async def test_turn_collects_then_confirms_and_advances(monkeypatch):
    from agents.xbuddy.agent import graph

    cfg = {"configurable": {"thread_id": "flow-1", "user_id": 1}}
    await graph.ainvoke({"messages": []}, cfg)  # new thread

    # Turn 1: user answers; agent summarizes; decision stays and saves a draft.
    fake(monkeypatch, reply_mod, "Summary: backend dev, 3 years, Python. Look right?")
    fake(monkeypatch, decision_mod, json.dumps({
        "router_directive": "stay", "should_save_content": True,
        "section_summary": "Backend dev, 3 years, Python"}))
    s1 = await graph.ainvoke({"messages": [HumanMessage("Backend dev, 3 yrs, Python")]}, cfg)

    assert s1["current_section"] == SectionID.BACKGROUND
    bg = s1["section_states"]["background"]
    assert bg.content.plain_text == "Backend dev, 3 years, Python"
    assert bg.confirmed_summary is None
    assert isinstance(s1["messages"][-1], AIMessage)

    # Turn 2: user confirms; agent hands off; decision says next.
    fake(monkeypatch, reply_mod, "Thanks! What job titles are you aiming for next?")
    fake(monkeypatch, decision_mod, json.dumps({
        "router_directive": "next", "is_satisfied": True, "should_save_content": True,
        "section_summary": "Backend dev, 3 years, Python",
        "covered_fields": get_section_template(SectionID.BACKGROUND).required_fields}))
    fake(monkeypatch, extraction_mod, json.dumps({
        "current_role": "Backend Developer", "years_experience": 3, "skills": ["Python"],
        "role_history": [], "education": None}))
    s2 = await graph.ainvoke({"messages": [HumanMessage("Yes, looks good")]}, cfg)

    assert s2["current_section"] == SectionID.TARGET_ROLE
    assert s2["section_states"]["background"].status == SectionStatus.DONE
    assert s2["section_states"]["background"].confirmed_summary == "Backend dev, 3 years, Python"
    assert s2["section_states"]["target_role"].status == SectionStatus.IN_PROGRESS
    assert s2["user_data"].profile.current_role == "Backend Developer"
    assert [m.content for m in s2["short_memory"]] == [
        "Thanks! What job titles are you aiming for next?"
    ]
    assert s2["router_directive"] == "stay"

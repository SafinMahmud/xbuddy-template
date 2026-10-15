"""PR 6 tests: /invoke, /stream, /history and /roadmap against the JobBuddy graph.

Runs the real FastAPI app with its SQLite checkpointer (as in production) and
fake models, so no API keys or network are needed.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage

from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy.agent import graph
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import SectionState
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import implementation as impl
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.prompts import get_section_template

SUMMARY_REPLY = "Summary: backend dev, 3 years, Python. Does this look right?"
STAY = json.dumps({"router_directive": "stay", "should_save_content": True,
                   "section_summary": "Backend dev, 3 years, Python"})
NEXT = json.dumps({"router_directive": "next", "is_satisfied": True, "should_save_content": True,
                   "section_summary": "Backend dev, 3 years, Python",
                   "covered_fields": get_section_template(SectionID.BACKGROUND).required_fields})


def set_models(monkeypatch, reply: str, decision: str):
    monkeypatch.setattr(reply_mod, "get_chat_model", lambda config=None: GenericFakeChatModel(
        messages=iter([AIMessage(reply)])))
    monkeypatch.setattr(decision_mod, "get_chat_model",
                        lambda config=None: FakeListChatModel(responses=[decision]))
    monkeypatch.setattr(extraction_mod, "get_chat_model", lambda config=None: FakeListChatModel(
        responses=['{"current_role": "Backend Developer", "skills": ["Python"]}']))


@pytest.fixture
def client(monkeypatch, tmp_path):
    from core.settings import DatabaseType, settings

    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.SQLITE)
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(settings, "USE_SUPABASE_REALTIME", False)
    monkeypatch.setattr(mu, "get_section_store", lambda: None)
    monkeypatch.setattr(impl, "get_roadmap_store", lambda: None)
    original = graph.checkpointer
    from service import app

    with TestClient(app) as c:
        yield c
    graph.checkpointer = original


def invoke(client, message, thread_id=None):
    body = {"message": message, "user_id": 7}
    if thread_id:
        body["thread_id"] = thread_id
    res = client.post("/invoke", json=body)
    assert res.status_code == 200, res.text
    return res.json()


def test_invoke_new_thread_then_confirm_advances(client, monkeypatch):
    set_models(monkeypatch, SUMMARY_REPLY, STAY)
    first = invoke(client, "Backend dev, 3 yrs, Python")

    assert first["user_id"] == 7
    assert first["output"]["type"] == "ai"
    assert first["output"]["content"] == SUMMARY_REPLY
    section = first["output"]["custom_data"]["section"]
    assert section["id"] == "background"
    assert section["name"] == "Background"
    assert section["completed_sections"] == 0
    assert section["total_sections"] == 5

    set_models(monkeypatch, "Thanks! What job titles are you aiming for next?", NEXT)
    second = invoke(client, "Yes, looks good", thread_id=first["thread_id"])

    assert second["thread_id"] == first["thread_id"]
    section = second["output"]["custom_data"]["section"]
    assert section["id"] == "target_role"
    assert section["completed_sections"] == 1
    assert section["roadmap_ready"] is False
    # The whole path, in order, for the frontend's progress display.
    assert [(s["id"], s["status"]) for s in section["sections"]] == [
        ("background", "done"),
        ("target_role", "in_progress"),
        ("skill_gap", "pending"),
        ("application_strategy", "pending"),
        ("interview_prep", "pending"),
    ]
    assert section["sections"][2]["name"] == "Skill Gap"


def test_history_restores_messages_and_progress(client, monkeypatch):
    set_models(monkeypatch, SUMMARY_REPLY, STAY)
    thread_id = invoke(client, "Backend dev, 3 yrs, Python")["thread_id"]

    res = client.post("/history", json={"thread_id": thread_id})
    assert res.status_code == 200
    body = res.json()
    assert [m["type"] for m in body["messages"]] == ["human", "ai"]
    assert body["messages"][0]["custom_data"]["section_id"] == "background"
    assert body["section"]["id"] == "background"
    assert body["section"]["status"] == "in_progress"


def test_history_unknown_thread_is_empty(client):
    res = client.post("/history", json={"thread_id": "does-not-exist"})
    assert res.status_code == 200
    assert res.json() == {"messages": [], "section": None}


def test_stream_sends_reply_tokens_but_not_the_decision(client, monkeypatch):
    set_models(monkeypatch, SUMMARY_REPLY, STAY)
    with client.stream("POST", "/stream", json={"message": "Backend dev", "user_id": 7}) as res:
        assert res.status_code == 200
        events = [line[len("data: "):] for line in res.iter_lines() if line.startswith("data: ")]

    parsed = [json.loads(e) for e in events if e != "[DONE]"]
    tokens = [e["content"] for e in parsed if e["type"] == "token"]
    assert len(tokens) > 1
    assert "".join(tokens) == SUMMARY_REPLY
    assert not any("router_directive" in t for t in tokens)  # decision JSON never streamed
    section = [e["content"] for e in parsed if e["type"] == "section"]
    assert section and section[-1]["id"] == "background"
    assert events[-1] == "[DONE]"


def test_stream_delivers_the_roadmap_in_the_turn_that_writes_it(client, monkeypatch):
    """The last confirmation adds two messages: the reply, then the roadmap."""
    import asyncio

    set_models(monkeypatch, SUMMARY_REPLY, STAY)
    thread_id = invoke(client, "Backend dev")["thread_id"]

    # Jump to the end: four sections confirmed, the fifth waiting for its "yes".
    states = {
        s.value: SectionState(section_id=s, status=SectionStatus.DONE, confirmed_summary="ok")
        for s in SectionID
    }
    states["interview_prep"] = SectionState(
        section_id=SectionID.INTERVIEW_PREP, status=SectionStatus.IN_PROGRESS
    )
    asyncio.run(graph.aupdate_state(
        {"configurable": {"thread_id": thread_id}},
        {"section_states": states, "current_section": SectionID.INTERVIEW_PREP,
         "context_packet": None},
    ))  # fmt: skip

    closing = "Thanks! I am now putting together your roadmap."
    confirm = json.dumps({
        "router_directive": "next", "is_satisfied": True, "should_save_content": True,
        "section_summary": "Behavioral and system design, weekly mock interviews",
        "covered_fields": get_section_template(SectionID.INTERVIEW_PREP).required_fields})
    set_models(monkeypatch, closing, confirm)

    def no_model(config=None):
        raise RuntimeError("roadmap model unavailable")  # the fallback roadmap is used

    monkeypatch.setattr(impl, "get_chat_model", no_model)

    body = {"message": "Yes, looks good", "user_id": 7, "thread_id": thread_id}
    with client.stream("POST", "/stream", json=body) as res:
        assert res.status_code == 200
        events = [line[len("data: "):] for line in res.iter_lines() if line.startswith("data: ")]

    parsed = [json.loads(e) for e in events if e != "[DONE]"]
    sent = [e["content"]["content"] for e in parsed if e["type"] == "message"]
    assert len(sent) == 2, sent
    assert sent[0] == closing
    assert sent[1].startswith("# Your Job Search Roadmap")
    assert "".join(e["content"] for e in parsed if e["type"] == "token") == closing
    section = [e["content"] for e in parsed if e["type"] == "section"][-1]
    assert section["roadmap_ready"] is True

    # A later turn sends its one new reply and never repeats the roadmap.
    set_models(monkeypatch, "Happy to help with that.", STAY)
    body["message"] = "Can you explain week 1?"
    with client.stream("POST", "/stream", json=body) as res:
        events = [line[len("data: "):] for line in res.iter_lines() if line.startswith("data: ")]
    sent = [json.loads(e)["content"]["content"] for e in events
            if e != "[DONE]" and json.loads(e)["type"] == "message"]
    assert sent == ["Happy to help with that."]


def test_roadmap_endpoint_before_and_after_completion(client, monkeypatch):
    set_models(monkeypatch, SUMMARY_REPLY, STAY)
    thread_id = invoke(client, "Backend dev")["thread_id"]

    res = client.get(f"/roadmap/xbuddy?thread_id={thread_id}&user_id=7").json()
    assert res["success"] is False
    assert "finish all five sections" in res["message"]

    import asyncio

    config = {"configurable": {"thread_id": thread_id}}
    done = {s.value: SectionState(section_id=s, status=SectionStatus.DONE) for s in SectionID}

    async def finish():
        await graph.aupdate_state(config, {"roadmap": "# Your Job Search Roadmap",
                                           "section_states": done})

    asyncio.run(finish())
    res = client.get(f"/roadmap/xbuddy?thread_id={thread_id}&user_id=7").json()
    assert res["success"] is True
    assert res["roadmap"] == res["business_plan"] == "# Your Job Search Roadmap"
    assert res["section"]["completed_sections"] == 5
    assert res["section"]["roadmap_ready"] is True


def test_unknown_agent_roadmap_is_404(client):
    assert client.get("/roadmap/nope?thread_id=x").status_code == 404

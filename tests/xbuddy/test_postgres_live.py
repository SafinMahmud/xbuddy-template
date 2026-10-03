"""Live Postgres durability tests. They need a real database, so they are opt-in.

    TEST_POSTGRES_URL=postgresql://user:password@host:5432/dbname \\
        uv run pytest tests/xbuddy/test_postgres_live.py -q

Skipped when TEST_POSTGRES_URL is not set. The database must use UTF8 encoding.
The tests run the real FastAPI app with the Postgres checkpointer and fake models,
and delete the threads they create.
"""

import json
import os
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from pydantic import SecretStr

import smoke_test
from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy.agent import graph
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import implementation as impl
from agents.xbuddy.nodes import memory_updater as mu

POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="TEST_POSTGRES_URL is not set")

REPLY = "Thanks. What job titles have you held before?"
STAY = json.dumps({"router_directive": "stay", "should_save_content": True,
                   "section_summary": "Backend dev, 3 years, Python"})


def _delete_threads(thread_ids):
    import psycopg

    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ANY(%s)", (list(thread_ids),))


@pytest.fixture
def start_service(monkeypatch):
    """Each `with start_service()` is one service lifetime with its own connection pool,
    so leaving it and entering it again is a restart against the same database."""
    from core.settings import DatabaseType, settings

    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.POSTGRES)
    monkeypatch.setattr(settings, "POSTGRES_URL", SecretStr(POSTGRES_URL))
    monkeypatch.setattr(settings, "USE_SUPABASE_REALTIME", False)
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(mu, "get_section_store", lambda: None)
    monkeypatch.setattr(impl, "get_roadmap_store", lambda: None)
    monkeypatch.setattr(reply_mod, "get_chat_model",
                        lambda config=None: GenericFakeChatModel(messages=iter([AIMessage(REPLY)])))
    monkeypatch.setattr(decision_mod, "get_chat_model",
                        lambda config=None: FakeListChatModel(responses=[STAY]))
    monkeypatch.setattr(extraction_mod, "get_chat_model",
                        lambda config=None: FakeListChatModel(responses=[STAY]))
    original = graph.checkpointer
    created: list[str] = []
    from service import app

    @contextmanager
    def start():
        with TestClient(app) as client:
            client.created = created
            yield client

    yield start
    graph.checkpointer = original
    _delete_threads(created)


def test_thread_survives_a_restart_on_postgres(start_service):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    with start_service() as before:
        assert isinstance(graph.checkpointer, AsyncPostgresSaver)
        assert before.get("/health").json()["storage"] == {"type": "postgres", "durable": True}
        first = smoke_test.run_smoke(before, expect="open")
        before.created.append(first.thread_id)
    assert first.passed, [c.name for c in first.checks if not c.ok]

    with start_service() as after:  # new process lifetime, new connection pool
        result = smoke_test.run_smoke(after, resume=first.thread_id)
        history = after.post("/history", json={"thread_id": first.thread_id}).json()

    assert result.passed, [(c.name, c.detail) for c in result.checks if not c.ok]
    assert [m["type"] for m in history["messages"]] == ["human", "ai"] * 3
    assert history["messages"][0]["content"] == smoke_test.FIRST_MESSAGE
    assert history["messages"][4]["content"] == smoke_test.RESUME_MESSAGE
    assert history["section"]["id"] == "background"


def test_section_progress_survives_a_restart_on_postgres(start_service):
    thread_id = "live-pg-draft"
    with start_service() as before:
        before.created.append(thread_id)
        res = before.post("/invoke", json={"message": "Backend dev, 3 years, Python",
                                           "user_id": 7, "thread_id": thread_id})
        assert res.status_code == 200, res.text

    with start_service() as after:
        roadmap = after.get(f"/roadmap/xbuddy?thread_id={thread_id}&user_id=7").json()
        history = after.post("/history", json={"thread_id": thread_id}).json()

    assert roadmap["success"] is False
    assert roadmap["section"]["id"] == "background"
    assert roadmap["section"]["status"] == "in_progress"
    assert [m["type"] for m in history["messages"]] == ["human", "ai"]


def test_threads_are_isolated_on_postgres(start_service):
    with start_service() as client:
        client.created.extend(["live-pg-a", "live-pg-b"])
        client.post("/invoke", json={"message": "I am thread A", "user_id": 1,
                                     "thread_id": "live-pg-a"})
        client.post("/invoke", json={"message": "I am thread B", "user_id": 2,
                                     "thread_id": "live-pg-b"})
        a = client.post("/history", json={"thread_id": "live-pg-a"}).json()
        b = client.post("/history", json={"thread_id": "live-pg-b"}).json()
        unknown = client.post("/history", json={"thread_id": "live-pg-never-used"}).json()

    assert [m["content"] for m in a["messages"] if m["type"] == "human"] == ["I am thread A"]
    assert [m["content"] for m in b["messages"] if m["type"] == "human"] == ["I am thread B"]
    assert unknown == {"messages": [], "section": None}

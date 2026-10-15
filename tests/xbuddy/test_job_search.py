"""Phase 8 tests: the search_jobs tool and the tool loop in the graph.

No test touches the network: job-tracker's Supabase REST API is replaced with an
httpx.MockTransport, and the chat model with a fake that can ask for a tool.
"""

import itertools
import json
import re

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGenerationChunk
from pydantic import Field, SecretStr

from agents.xbuddy import job_search
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.graph.routes import route_after_reply
from agents.xbuddy.job_search import (
    JobSearchError,
    build_params,
    fetch_jobs,
    location_term,
    quoted,
    search_jobs,
    search_words,
)
from agents.xbuddy.models import default_state
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.nodes.generate_reply import (
    FALLBACK_REPLY,
    MAX_TOOL_ROUNDS,
    TOOL_LIMIT_REPLY,
    generate_reply_node,
)
from agents.xbuddy.nodes.run_tools import run_tools_node
from agents.xbuddy.tools import build_context_packet
from core.settings import settings

CONFIG = {"configurable": {"thread_id": "t", "user_id": 1}}

POSTING_URL = "https://boards.greenhouse.io/maple/jobs/101"
TRACKER_OK = [
    {
        "title": "Backend Developer (Python)",
        "company": "Maple Systems",
        "location": "Toronto, Ontario",
        "url": POSTING_URL,
        "ats_platform": "greenhouse",
        "date_discovered": "2026-10-01T09:30:00+00:00",
        "description": "We are looking for a backend developer with Django and...",
    },
    {"title": "No link, so it is dropped", "company": "Ghost Inc", "url": None},
]
KEY = "anon-key-456"


# --- helpers ---------------------------------------------------------------------------------


@pytest.fixture
def tracker(monkeypatch):
    """Configure job-tracker settings and return a function that installs a fake server."""
    monkeypatch.setattr(settings, "JOB_TRACKER_SUPABASE_URL", "https://tracker.supabase.co/")
    monkeypatch.setattr(settings, "JOB_TRACKER_SUPABASE_KEY", SecretStr(KEY))
    monkeypatch.setattr(job_search, "BACKOFF_SECONDS", (0, 0))  # no real waiting in tests

    def install(*responses):
        """Each call to the fake server returns the next response (the last one repeats)."""
        requests: list[httpx.Request] = []
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(job_search, "_TRANSPORT", httpx.MockTransport(handler))
        return requests

    return install


class ToolCallingFake(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools and records each prompt it receives."""

    prompts: list = Field(default_factory=list)
    bound: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound.append([t.name for t in tools])
        return self

    async def ainvoke(self, input, config=None, **kwargs):
        self.prompts.append(list(input))
        return await super().ainvoke(input, config, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Stream a tool call the way real providers do: as tool_call_chunks.

        The stock fake can only stream text, so under /stream a tool request
        would be lost and the test would pass for the wrong reason.
        """
        message = next(self.messages)
        if message.tool_calls:
            chunks = [
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(message.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        self.messages = itertools.chain([message], self.messages)
        yield from super()._stream(messages, stop, run_manager, **kwargs)


def tool_request(call_id="call_1", **args) -> AIMessage:
    args = args or {"query": "backend developer", "location": "Toronto"}
    return AIMessage(content="", tool_calls=[{"name": "search_jobs", "args": args, "id": call_id}])


def use_reply_model(monkeypatch, *messages) -> ToolCallingFake:
    model = ToolCallingFake(messages=iter(messages))
    monkeypatch.setattr(reply_mod, "get_chat_model", lambda config=None: model)
    return model


def skill_gap_state(**extra) -> dict:
    state = {"messages": [], "thread_id": "t", "user_id": 1, **default_state()}
    for sid in (SectionID.BACKGROUND, SectionID.TARGET_ROLE):
        state["section_states"][sid.value] = state["section_states"][sid.value].model_copy(
            update={"status": SectionStatus.DONE, "confirmed_summary": "confirmed"}
        )
    state["section_states"]["skill_gap"] = state["section_states"]["skill_gap"].model_copy(
        update={"status": SectionStatus.IN_PROGRESS}
    )
    state["current_section"] = SectionID.SKILL_GAP
    state["context_packet"] = build_context_packet(SectionID.SKILL_GAP, state["section_states"])
    state.update(extra)
    return state


# --- turning a request into a safe query -----------------------------------------------------


@pytest.mark.parametrize(
    ("query", "words"),
    [
        ("backend developer", ["backend", "developer"]),
        ("Python developer jobs in Toronto", ["python", "developer", "toronto"]),
        ("node.js engineer", ["node", "js", "engineer"]),
        ("C# / C++ roles", ["c#", "c++"]),
        ("a b c d e f g", ["b", "c", "d", "e"]),  # at most four words, filler dropped
    ],
)
def test_search_words_keep_letters_digits_plus_and_hash_only(query, words):
    assert search_words(query) == words


def test_model_text_cannot_change_the_shape_of_the_filter():
    words = search_words("backend),title.neq.x,(or")
    params = build_params(words, location_term("Toronto),(x"), 5)
    assert params["and"] == (
        '(or(title.ilike."*backend*",company.ilike."*backend*"),'
        'or(title.ilike."*title*",company.ilike."*title*"),'
        'or(title.ilike."*neq*",company.ilike."*neq*"),'
        'or(title.ilike."*x*",company.ilike."*x*"))'
    )
    assert params["location"] == "ilike.*toronto*"


def test_quoting_escapes_quotes_and_backslashes():
    assert quoted('*c++*') == '"*c++*"'
    assert quoted('a"b\\c') == '"a\\"b\\\\c"'


# The only shapes a filter may ever take, whatever the model sends.
SAFE_AND = re.compile(
    r'\((?:or\(title\.ilike\."\*([a-z0-9+#]+)\*",company\.ilike\."\*\1\*"\),?){1,4}\)'
)
SAFE_LOCATION = re.compile(r"ilike\.\*[a-z ]+\*")
HOSTILE = [
    'backend", company.neq.x',  # break out of the quotes
    "backend),or(title.neq.zzz",  # close the group and add a clause
    "python\\\" developer",  # backslash before a quote
    "dev\n&limit=1000&select=*",  # extra query parameters
    "%2C%28or%29 %22",  # pre-encoded reserved characters
    "' OR 1=1 --",  # SQL-style
    "*",  # wildcard only
    "c++ c# .net node.js",  # legitimate symbols
    "développeur logiciel",  # accents
    "a" * 500,  # very long
]


@pytest.mark.parametrize("hostile", HOSTILE)
def test_hostile_text_only_ever_produces_the_safe_filter_shape(hostile):
    params = build_params(search_words(hostile), location_term(hostile), 5)

    assert set(params) <= {"select", "order", "limit", "and", "location"}
    assert params["select"] == job_search.COLUMNS and params["limit"] == "5"
    if "and" in params:
        assert SAFE_AND.fullmatch(params["and"]), params["and"]
    if "location" in params:
        assert SAFE_LOCATION.fullmatch(params["location"]), params["location"]


@pytest.mark.asyncio
async def test_plus_and_hash_are_percent_encoded_on_the_wire(tracker):
    requests = tracker(httpx.Response(200, json=TRACKER_OK))

    await fetch_jobs("c++ c# developer", "Toronto")

    raw_query = requests[0].url.query.decode()
    assert "%2B%2B" in raw_query and "%23" in raw_query
    assert "+" not in raw_query and "#" not in raw_query
    assert requests[0].url.params["and"].count('"*') == 6  # every value quoted


@pytest.mark.parametrize(
    ("given", "term"), [("Toronto, ON", "toronto"), ("North York", "north york"), ("", "")]
)
def test_location_uses_the_city_before_the_comma(given, term):
    assert location_term(given) == term


# --- the job-tracker client ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_queries_the_jobs_table_and_returns_clean_postings(tracker):
    requests = tracker(httpx.Response(200, json=TRACKER_OK))

    postings = await fetch_jobs("backend developer", "Toronto, ON")

    assert len(requests) == 1
    sent = requests[0]
    assert str(sent.url).startswith("https://tracker.supabase.co/rest/v1/jobs?")
    assert sent.headers["apikey"] == KEY
    assert "authorization" not in sent.headers  # the anon key only, no user login
    assert sent.url.params["order"] == "date_discovered.desc"
    assert sent.url.params["limit"] == "5"
    assert sent.url.params["location"] == "ilike.*toronto*"
    assert "description" not in sent.url.params["select"]  # never fetched
    assert [p.model_dump() for p in postings] == [
        {
            "title": "Backend Developer (Python)",
            "company": "Maple Systems",
            "location": "Toronto, Ontario",
            "posted": "2026-10-01",
            "board": "greenhouse",
            "url": POSTING_URL,
        }
    ]


@pytest.mark.asyncio
async def test_transient_failures_are_retried_until_one_succeeds(tracker):
    requests = tracker(
        httpx.Response(503),
        httpx.ConnectTimeout("slow"),
        httpx.Response(200, json=TRACKER_OK),
    )

    postings = await fetch_jobs("backend developer")

    assert len(requests) == 3
    assert postings[0].company == "Maple Systems"
    assert "location" not in requests[0].url.params  # no location: all of the coverage area


@pytest.mark.asyncio
async def test_retries_stop_after_the_attempt_limit(tracker):
    requests = tracker(httpx.Response(503))

    with pytest.raises(JobSearchError) as error:
        await fetch_jobs("backend developer")

    assert len(requests) == job_search.MAX_ATTEMPTS
    assert error.value.retryable


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 400])
async def test_permanent_failures_are_not_retried(tracker, status):
    requests = tracker(httpx.Response(status))

    with pytest.raises(JobSearchError) as error:
        await fetch_jobs("backend developer")

    assert len(requests) == 1
    assert not error.value.retryable


@pytest.mark.asyncio
async def test_missing_key_fails_without_calling_the_api(tracker, monkeypatch):
    requests = tracker(httpx.Response(200, json=TRACKER_OK))
    monkeypatch.setattr(settings, "JOB_TRACKER_SUPABASE_KEY", None)

    with pytest.raises(JobSearchError, match="not set up"):
        await fetch_jobs("backend developer")

    assert requests == []


@pytest.mark.asyncio
async def test_a_query_of_only_filler_words_is_refused_without_calling_the_api(tracker):
    requests = tracker(httpx.Response(200, json=TRACKER_OK))

    with pytest.raises(JobSearchError, match="job title or skill"):
        await fetch_jobs("jobs in the")

    assert requests == []


# --- the tool the model sees -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_returns_results_with_source_and_no_description(tracker):
    tracker(httpx.Response(200, json=TRACKER_OK))

    raw = await search_jobs.ainvoke({"query": "backend developer", "location": "Toronto"})

    result = json.loads(raw)
    assert result["status"] == "ok"
    assert "company career boards" in result["source"]
    assert result["covers"] == "software roles in Ontario and remote Canada"
    assert result["results"][0]["url"] == POSTING_URL
    assert "description" not in raw and "Django" not in raw


@pytest.mark.asyncio
async def test_tool_reports_failure_as_data_and_never_raises(tracker):
    tracker(httpx.ReadTimeout("no answer"))

    result = json.loads(await search_jobs.ainvoke({"query": "backend developer"}))

    assert result == {
        "status": "error",
        "message": "The job search service took too long to answer.",
    }


@pytest.mark.asyncio
async def test_tool_never_leaks_the_key_in_an_error(tracker):
    tracker(httpx.Response(401, text=f"invalid api key {KEY}"))

    raw = await search_jobs.ainvoke({"query": "backend developer"})

    assert json.loads(raw)["status"] == "error"
    assert KEY not in raw


@pytest.mark.asyncio
async def test_tool_says_no_results_and_what_it_covers(tracker):
    tracker(httpx.Response(200, json=[]))

    result = json.loads(await search_jobs.ainvoke({"query": "underwater basket weaver"}))

    assert result["status"] == "no_results"
    assert result["covers"] == "software roles in Ontario and remote Canada"


def test_tool_schema_is_plain_strings_with_only_the_query_required():
    schema = search_jobs.args_schema.model_json_schema()
    assert schema["required"] == ["query"]
    assert {k: v["type"] for k, v in schema["properties"].items()} == {
        "query": "string",
        "location": "string",
    }


# --- generate_reply and the tools node -------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tool_request_is_parked_in_scratch_and_is_not_a_reply(monkeypatch):
    model = use_reply_model(monkeypatch, tool_request())
    user = HumanMessage("I have no postings. Can you find some?")

    result = await generate_reply_node(skill_gap_state(messages=[user]), CONFIG)

    assert model.bound == [["search_jobs", "read_job_posting"]]
    assert "JOB SEARCH TOOL" in model.prompts[0][0].content
    assert set(result) == {"tool_scratch"}  # no message, no short memory change
    assert result["tool_scratch"][0].tool_calls[0]["name"] == "search_jobs"
    assert route_after_reply(result) == "tools"


@pytest.mark.asyncio
async def test_tools_node_answers_every_call_including_broken_ones(tracker):
    tracker(httpx.Response(200, json=TRACKER_OK))
    request = AIMessage(content="", tool_calls=[
        {"name": "search_jobs", "args": {"query": "backend developer"}, "id": "a"},
        {"name": "search_jobs", "args": {"location": "Toronto"}, "id": "b"},  # no query
        {"name": "delete_everything", "args": {}, "id": "c"},
    ])  # fmt: skip

    result = await run_tools_node({"tool_scratch": [request]}, CONFIG)

    answers = result["tool_scratch"][1:]
    assert all(isinstance(m, ToolMessage) for m in answers)
    assert [m.tool_call_id for m in answers] == ["a", "b", "c"]
    assert [json.loads(m.content)["status"] for m in answers] == ["ok", "error", "error"]


@pytest.mark.asyncio
async def test_reply_after_the_tool_uses_results_and_clears_scratch(monkeypatch):
    model = use_reply_model(monkeypatch, AIMessage("Here is one: Backend Developer at Maple."))
    user = HumanMessage("Can you find some?")
    scratch = [tool_request(), ToolMessage(content='{"status": "ok"}', tool_call_id="call_1")]

    result = await generate_reply_node(
        skill_gap_state(messages=[user], tool_scratch=scratch), CONFIG
    )

    sent = model.prompts[0]
    assert isinstance(sent[0], SystemMessage) and sent[1] is user
    assert sent[-2:] == scratch  # the model sees its request and the result
    assert result["messages"][0].content == "Here is one: Backend Developer at Maple."
    assert result["tool_scratch"] == []
    assert [type(m) for m in result["short_memory"]] == [HumanMessage, AIMessage]
    assert route_after_reply(result) == "generate_decision"


@pytest.mark.asyncio
async def test_the_model_cannot_search_forever(monkeypatch):
    use_reply_model(monkeypatch, tool_request("call_9"))
    scratch = []
    for i in range(MAX_TOOL_ROUNDS):
        scratch += [tool_request(f"c{i}"), ToolMessage(content="{}", tool_call_id=f"c{i}")]

    result = await generate_reply_node(
        skill_gap_state(messages=[HumanMessage("more")], tool_scratch=scratch), CONFIG
    )

    assert result["messages"][0].content == TOOL_LIMIT_REPLY
    assert result["tool_scratch"] == []


@pytest.mark.asyncio
async def test_a_failing_tool_call_falls_back_to_a_reply_without_tools(monkeypatch):
    class RejectsToolCalls(ToolCallingFake):
        async def ainvoke(self, input, config=None, **kwargs):
            if "JOB SEARCH TOOL" in input[0].content:
                self.prompts.append(list(input))
                raise ValueError("tool_use_failed")
            return await super().ainvoke(input, config, **kwargs)

    model = RejectsToolCalls(messages=iter([AIMessage("Search is down. Please paste one.")]))
    monkeypatch.setattr(reply_mod, "get_chat_model", lambda config=None: model)

    result = await generate_reply_node(
        skill_gap_state(messages=[HumanMessage("find me jobs")]), CONFIG
    )

    assert result["messages"][0].content == "Search is down. Please paste one."
    assert "unavailable right now" in model.prompts[-1][0].content
    assert "error_count" not in result


@pytest.mark.asyncio
async def test_sections_that_do_not_need_search_get_no_tool(monkeypatch):
    model = use_reply_model(monkeypatch, AIMessage("What is your current job title?"))
    state = {"messages": [HumanMessage("hi")], "thread_id": "t", "user_id": 1, **default_state()}

    await generate_reply_node(state, CONFIG)  # Background section

    assert model.bound == []
    assert "JOB SEARCH TOOL" not in model.prompts[0][0].content


@pytest.mark.asyncio
async def test_a_model_without_tool_support_still_replies(monkeypatch):
    model = FakeListChatModel(responses=["Please paste a posting."])  # bind_tools not implemented
    monkeypatch.setattr(reply_mod, "get_chat_model", lambda config=None: model)

    result = await generate_reply_node(skill_gap_state(messages=[HumanMessage("hi")]), CONFIG)

    assert result["messages"][0].content == "Please paste a posting."


# --- the whole graph -------------------------------------------------------------------------


async def _thread_in_skill_gap(graph, cfg):
    await graph.ainvoke({"messages": []}, cfg)
    state = skill_gap_state()
    await graph.aupdate_state(
        cfg,
        {k: state[k] for k in ("section_states", "current_section", "context_packet")}
        | {"router_directive": "stay"},
    )


def _decision_stays(monkeypatch):
    monkeypatch.setattr(mu, "get_section_store", lambda: None)
    monkeypatch.setattr(
        decision_mod,
        "get_chat_model",
        lambda config=None: FakeListChatModel(responses=['{"router_directive": "stay"}']),
    )


@pytest.mark.asyncio
async def test_graph_turn_searches_then_replies_and_keeps_the_transcript_clean(
    tracker, monkeypatch
):
    from agents.xbuddy.agent import graph

    requests = tracker(httpx.Response(200, json=TRACKER_OK))
    _decision_stays(monkeypatch)
    model = use_reply_model(
        monkeypatch,
        tool_request(query="backend developer", location="Toronto"),
        AIMessage("I found 1 opening: Backend Developer (Python) at Maple Systems."),
    )
    cfg = {"configurable": {"thread_id": "tools-1", "user_id": 1}}
    await _thread_in_skill_gap(graph, cfg)

    out = await graph.ainvoke({"messages": [HumanMessage("No postings yet. Find some?")]}, cfg)

    assert len(requests) == 1 and requests[0].url.params["location"] == "ilike.*toronto*"
    tool_result = model.prompts[1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert "Maple Systems" in tool_result.content
    # The user-facing transcript holds the question and the answer, nothing else.
    assert [type(m) for m in out["messages"]] == [HumanMessage, AIMessage]
    assert out["messages"][-1].content.startswith("I found 1 opening")
    assert not out["messages"][-1].tool_calls
    assert out["tool_scratch"] == []
    assert out["current_section"] == SectionID.SKILL_GAP


@pytest.mark.asyncio
async def test_graph_turn_survives_the_job_api_being_down(tracker, monkeypatch):
    from agents.xbuddy.agent import graph

    requests = tracker(httpx.Response(503))
    _decision_stays(monkeypatch)
    model = use_reply_model(
        monkeypatch,
        tool_request(),
        AIMessage("The job search is busy right now. Could you paste a posting instead?"),
    )
    cfg = {"configurable": {"thread_id": "tools-2", "user_id": 1}}
    await _thread_in_skill_gap(graph, cfg)

    out = await graph.ainvoke({"messages": [HumanMessage("Find me some postings")]}, cfg)

    assert len(requests) == job_search.MAX_ATTEMPTS  # retried, then gave up
    told = json.loads(model.prompts[1][-1].content)
    assert told["status"] == "error" and "busy" in told["message"]
    assert out["messages"][-1].content.startswith("The job search is busy")
    assert out["messages"][-1].content != FALLBACK_REPLY
    assert out.get("error_count", 0) == 0  # a failed search is not a failed turn


@pytest.mark.asyncio
async def test_leftover_tool_traffic_from_a_dead_turn_is_dropped(monkeypatch):
    from agents.xbuddy.agent import graph

    _decision_stays(monkeypatch)
    model = use_reply_model(monkeypatch, AIMessage("Please paste a posting."))
    cfg = {"configurable": {"thread_id": "tools-3", "user_id": 1}}
    await _thread_in_skill_gap(graph, cfg)
    await graph.aupdate_state(cfg, {"tool_scratch": [tool_request("stale")]})

    out = await graph.ainvoke({"messages": [HumanMessage("hello again")]}, cfg)

    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in model.prompts[0])
    assert out["messages"][-1].content == "Please paste a posting."


# --- through the API -------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from agents.xbuddy.agent import graph
    from core.settings import DatabaseType

    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.SQLITE)
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(settings, "USE_SUPABASE_REALTIME", False)
    original = graph.checkpointer
    from service import app

    with TestClient(app) as c:
        yield c
    graph.checkpointer = original


def test_stream_shows_the_answer_and_never_the_tool_traffic(client, tracker, monkeypatch):
    import asyncio

    from agents.xbuddy.agent import graph

    requests = tracker(httpx.Response(200, json=TRACKER_OK))
    _decision_stays(monkeypatch)
    answer = "I found one opening: Backend Developer (Python) at Maple Systems."
    use_reply_model(monkeypatch, AIMessage("What is your current job title?"))
    thread_id = client.post("/invoke", json={"message": "hi", "user_id": 7}).json()["thread_id"]
    state = skill_gap_state()
    asyncio.run(graph.aupdate_state(
        {"configurable": {"thread_id": thread_id}},
        {k: state[k] for k in ("section_states", "current_section", "context_packet")},
    ))  # fmt: skip

    use_reply_model(monkeypatch, tool_request(), AIMessage(answer))
    body = {"message": "Find me some postings", "user_id": 7, "thread_id": thread_id}
    with client.stream("POST", "/stream", json=body) as res:
        assert res.status_code == 200
        raw = [line[len("data: "):] for line in res.iter_lines() if line.startswith("data: ")]

    assert len(requests) == 1  # the search really ran while streaming
    events = [json.loads(e) for e in raw if e != "[DONE]"]
    assert "".join(e["content"] for e in events if e["type"] == "token") == answer
    messages = [e["content"] for e in events if e["type"] == "message"]
    assert [m["type"] for m in messages] == ["ai"] and messages[0]["content"] == answer
    assert not any(e["type"] == "error" for e in events)
    assert "greenhouse" not in "\n".join(raw)  # the raw tool result is never sent

    history = client.post("/history", json={"thread_id": thread_id}).json()
    assert [m["type"] for m in history["messages"]] == ["human", "ai", "human", "ai"]

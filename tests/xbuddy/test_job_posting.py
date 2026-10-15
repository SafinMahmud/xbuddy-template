"""PR 11 tests: read_job_posting, and chosen postings flowing into Skill Gap.

No network: job-tracker and Workday are replaced with an httpx.MockTransport that
answers by host, and the chat model with a fake that can ask for tools.
"""

import json

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field, SecretStr

# Shared fakes and state builders from the PR 8 tests.
from test_job_search import (
    _decision_stays,
    _thread_in_skill_gap,
    skill_gap_state,
    use_reply_model,
)

from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy import job_search
from agents.xbuddy.enums import SectionID
from agents.xbuddy.job_posting import (
    MAX_POSTINGS,
    MAX_TEXT,
    clean_link,
    html_to_text,
    read_job_posting,
    remember_postings,
)
from agents.xbuddy.nodes.generate_reply import generate_reply_node
from agents.xbuddy.nodes.run_tools import run_tools_node
from agents.xbuddy.prompts import get_section_template
from core.settings import settings

CONFIG = {"configurable": {"thread_id": "t", "user_id": 1}}
GREENHOUSE = "https://job-boards.greenhouse.io/braze/jobs/8226558"
WORKDAY = (
    "https://td.wd3.myworkdayjobs.com/en-US/TD_Bank_Careers/job/Toronto-Ontario/"
    "Mainframe-Software-Engineer-II_R_1512837"
)
GREENHOUSE_ROW = {
    "title": "Senior Software Engineer I, Email",
    "company": "Braze",
    "location": "Toronto",
    "url": GREENHOUSE,
    "ats_platform": "greenhouse",
    "description": "<p>You will build email services.</p><ul><li>5+ years of Go or Java</li>"
    "<li>Experience with Kafka</li></ul>",
}
WORKDAY_ROW = {
    "title": "Mainframe Software Engineer II",
    "company": "TD",
    "location": "Toronto, ON",
    "url": WORKDAY,
    "ats_platform": "workday",
    "description": "",
}
WORKDAY_DETAIL = {
    "jobPostingInfo": {
        "title": "Mainframe Software Engineer II",
        "jobDescription": "<p><b>What you need</b></p><ul><li>3+ years of COBOL</li>"
        "<li>DB2 &amp; CICS</li></ul>",
    }
}


@pytest.fixture
def servers(monkeypatch):
    """Fake job-tracker and Workday. Each takes a list of responses (last one repeats)."""
    monkeypatch.setattr(settings, "JOB_TRACKER_SUPABASE_URL", "https://tracker.supabase.co")
    monkeypatch.setattr(settings, "JOB_TRACKER_SUPABASE_KEY", SecretStr("anon-key"))
    monkeypatch.setattr(job_search, "BACKOFF_SECONDS", (0, 0))
    seen: list[httpx.Request] = []
    plans: dict[str, list] = {"tracker": [httpx.Response(200, json=[])], "workday": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        host = request.url.host
        key = "tracker" if host == "tracker.supabase.co" else "workday"
        if key == "workday" and not host.endswith(".myworkdayjobs.com"):
            raise AssertionError(f"unexpected host {host}")
        queue = plans[key]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(job_search, "_TRANSPORT", httpx.MockTransport(handler))

    def plan(tracker=None, workday=None):
        if tracker is not None:
            plans["tracker"] = list(tracker)
        if workday is not None:
            plans["workday"] = list(workday)
        return seen

    return plan


async def read(link: str) -> dict:
    return json.loads(await read_job_posting.ainvoke({"link": link}))


# --- helpers -------------------------------------------------------------------------------


def test_html_to_text_keeps_list_items_and_decodes_entities():
    text = html_to_text(WORKDAY_DETAIL["jobPostingInfo"]["jobDescription"])
    assert text == "What you need\n- 3+ years of COBOL\n- DB2 & CICS"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (f"<{GREENHOUSE}>", GREENHOUSE),
        (f"  {GREENHOUSE} ", GREENHOUSE),
        ("number 2", None),
        ("ftp://example.com/job", None),
        ("javascript:alert(1)", None),
        ("https://" + "a" * 600, None),
    ],
)
def test_clean_link_accepts_only_http_links(given, expected):
    assert clean_link(given) == expected


def test_remember_postings_keeps_one_per_link_and_the_newest_three():
    posts = [{"url": f"u{i}", "label": str(i), "text": "t"} for i in range(5)]
    kept = remember_postings(posts[:2], posts[2:])
    assert [p["url"] for p in kept] == ["u2", "u3", "u4"]
    again = remember_postings(kept, [{"url": "u2", "label": "new", "text": "t"}])
    assert [p["url"] for p in again] == ["u3", "u4", "u2"]
    assert len(again) == MAX_POSTINGS


# --- the tool ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_job_from_the_search_is_read_from_job_tracker(servers):
    seen = servers(tracker=[httpx.Response(200, json=[GREENHOUSE_ROW])])

    result = await read(GREENHOUSE)

    assert result["status"] == "ok"
    posting = result["posting"]
    assert posting["label"] == "Braze - Senior Software Engineer I, Email"
    assert posting["url"] == GREENHOUSE
    assert "- 5+ years of Go or Java" in posting["text"] and "<li>" not in posting["text"]
    assert posting["truncated"] is False
    assert len(seen) == 1
    assert seen[0].url.params["url"] == f"eq.{GREENHOUSE}"
    assert "description" in seen[0].url.params["select"]
    assert seen[0].headers["apikey"] == "anon-key"


@pytest.mark.asyncio
async def test_a_link_with_tracking_parameters_is_found_without_them(servers):
    seen = servers(tracker=[httpx.Response(200, json=[]), httpx.Response(200, json=[GREENHOUSE_ROW])])

    result = await read(GREENHOUSE + "?utm_source=linkedin&utm_medium=feed")

    assert result["status"] == "ok"
    assert [r.url.params["url"] for r in seen] == [
        f"eq.{GREENHOUSE}?utm_source=linkedin&utm_medium=feed",
        f"eq.{GREENHOUSE}",
    ]


@pytest.mark.asyncio
async def test_long_postings_are_cut_and_marked(servers):
    long_row = {**GREENHOUSE_ROW, "description": "x" * (MAX_TEXT + 500)}
    servers(tracker=[httpx.Response(200, json=[long_row])])

    posting = (await read(GREENHOUSE))["posting"]

    assert len(posting["text"]) == MAX_TEXT and posting["truncated"] is True


@pytest.mark.asyncio
async def test_a_workday_job_gets_its_text_from_workday(servers):
    seen = servers(
        tracker=[httpx.Response(200, json=[WORKDAY_ROW])],
        workday=[httpx.Response(200, json=WORKDAY_DETAIL)],
    )

    result = await read(WORKDAY)

    assert result["status"] == "ok"
    assert result["posting"]["text"].startswith("What you need\n- 3+ years of COBOL")
    detail = seen[1]
    assert detail.url.host == "td.wd3.myworkdayjobs.com"
    assert detail.url.path == (
        "/wday/cxs/td/TD_Bank_Careers/job/Toronto-Ontario/Mainframe-Software-Engineer-II_R_1512837"
    )


@pytest.mark.asyncio
async def test_workday_failures_are_retried_then_explained(servers):
    seen = servers(
        tracker=[httpx.Response(200, json=[WORKDAY_ROW])],
        workday=[httpx.Response(503)],
    )

    result = await read(WORKDAY)

    assert result["status"] == "no_text"
    assert "Workday is busy" in result["message"] and "paste" in result["message"]
    assert len(seen) == 1 + job_search.MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_a_stored_workday_link_of_the_wrong_shape_is_never_fetched(servers):
    odd = {**WORKDAY_ROW, "url": "https://evil.example.com/job/1"}
    seen = servers(tracker=[httpx.Response(200, json=[odd])])

    result = await read(WORKDAY)

    assert result["status"] == "no_text"
    assert len(seen) == 1  # only the job-tracker lookup, no fetch to the odd host


@pytest.mark.asyncio
async def test_a_job_without_any_stored_text_asks_for_a_paste(servers):
    seen = servers(tracker=[httpx.Response(200, json=[{**GREENHOUSE_ROW, "description": ""}])])

    result = await read(GREENHOUSE)

    assert result["status"] == "no_text" and "paste" in result["message"]
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_a_link_job_tracker_never_saw_is_not_fetched(servers):
    seen = servers(tracker=[httpx.Response(200, json=[])])

    result = await read("https://jobs.bell.ca/ca/en/job/BECACA433042EXTERNALENCA/AI-Engineer")

    assert result == {
        "status": "not_found",
        "message": "I can only read jobs from my own search. Please paste the posting text.",
    }
    assert all(r.url.host == "tracker.supabase.co" for r in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "link",
    ["https://www.linkedin.com/jobs/view/4474781558/", "https://ca.indeed.com/viewjob?jk=1"],
)
async def test_linkedin_and_indeed_links_are_refused_without_any_request(servers, link):
    seen = servers()

    result = await read(link)

    assert result["status"] == "unsupported" and "paste" in result["message"]
    assert seen == []


@pytest.mark.asyncio
async def test_text_that_is_not_a_link_is_refused_without_any_request(servers):
    seen = servers()

    result = await read("number 2")

    assert result["status"] == "error" and seen == []


@pytest.mark.asyncio
async def test_a_hostile_link_is_sent_as_one_encoded_filter_value(servers):
    seen = servers(tracker=[httpx.Response(200, json=[])])

    await read("https://x.example.com/job&select=*&or=(url.neq.x)")

    params = seen[0].url.params
    assert set(params) == {"select", "url", "limit"}
    assert params["select"] == "title,company,location,url,ats_platform,description"
    assert params["url"] == "eq.https://x.example.com/job&select=*&or=(url.neq.x)"


@pytest.mark.asyncio
async def test_job_tracker_down_is_explained_not_raised(servers):
    servers(tracker=[httpx.ConnectError("down")])

    result = await read(GREENHOUSE)

    assert result["status"] == "no_text" and "could not be reached" in result["message"]


# --- the tools node and the Skill Gap section ----------------------------------------------


def read_request(*links: str) -> AIMessage:
    calls = [
        {"name": "read_job_posting", "args": {"link": link}, "id": f"r{i}"}
        for i, link in enumerate(links)
    ]
    return AIMessage(content="", tool_calls=calls)


@pytest.mark.asyncio
async def test_tools_node_keeps_postings_it_read_and_skips_failures(servers):
    servers(tracker=[httpx.Response(200, json=[GREENHOUSE_ROW]), httpx.Response(200, json=[])])
    request = read_request(GREENHOUSE, "https://jobs.bell.ca/job/1")

    result = await run_tools_node({"tool_scratch": [request], "fetched_postings": []}, CONFIG)

    assert [json.loads(m.content)["status"] for m in result["tool_scratch"][1:]] == [
        "ok",
        "not_found",
    ]
    assert [p["label"] for p in result["fetched_postings"]] == [
        "Braze - Senior Software Engineer I, Email"
    ]


@pytest.mark.asyncio
async def test_tools_node_leaves_postings_alone_when_nothing_was_read(servers):
    servers(tracker=[httpx.Response(200, json=[])])

    result = await run_tools_node({"tool_scratch": [read_request(GREENHOUSE)]}, CONFIG)

    assert "fetched_postings" not in result


CHOSEN = {
    "label": "Braze - Senior Software Engineer I, Email",
    "url": GREENHOUSE,
    "text": "5+ years of Go or Java. Experience with Kafka.",
}


@pytest.mark.asyncio
async def test_skill_gap_replies_see_the_chosen_postings(monkeypatch):
    model = use_reply_model(monkeypatch, AIMessage("Braze asks for Kafka."))

    await generate_reply_node(
        skill_gap_state(messages=[HumanMessage("what am I missing?")], fetched_postings=[CHOSEN]),
        CONFIG,
    )

    system = model.prompts[0][0].content
    assert "POSTINGS THE USER CHOSE" in system
    assert "=== Braze - Senior Software Engineer I, Email" in system
    assert "Experience with Kafka." in system


@pytest.mark.asyncio
async def test_other_sections_do_not_carry_the_postings(monkeypatch):
    from agents.xbuddy.models import default_state

    model = use_reply_model(monkeypatch, AIMessage("What is your current title?"))
    state = {
        "messages": [HumanMessage("hi")], "thread_id": "t", "user_id": 1,
        **default_state(), "fetched_postings": [CHOSEN],
    }  # fmt: skip

    await generate_reply_node(state, CONFIG)  # Background

    assert "POSTINGS THE USER CHOSE" not in model.prompts[0][0].content


class RecordingExtraction(FakeListChatModel):
    prompts: list = Field(default_factory=list)

    async def ainvoke(self, input, config=None, **kwargs):
        self.prompts.append(list(input))
        return await super().ainvoke(input, config, **kwargs)


@pytest.mark.asyncio
async def test_extraction_quotes_chosen_postings_as_supplied_sources(monkeypatch):
    from agents.xbuddy.models import JobBuddyData

    answer = {
        "job_sources": [{"label": CHOSEN["label"], "url": GREENHOUSE}],
        "requirements": [{
            "name": "Kafka", "priority": "must_have",
            "evidence": [{"source_label": CHOSEN["label"], "quote": "Experience with Kafka"}],
        }],
    }  # fmt: skip
    model = RecordingExtraction(responses=[json.dumps(answer)])
    monkeypatch.setattr(extraction_mod, "get_chat_model", lambda config=None: model)

    data = await extraction_mod.extract_section_data(
        SectionID.SKILL_GAP,
        [HumanMessage("use number 1"), AIMessage("Braze asks for Kafka.")],
        "Gap: Kafka. Action: build a Kafka project.",
        JobBuddyData(),
        CONFIG,
        postings=[CHOSEN],
    )

    sent = model.prompts[0][1].content
    assert sent.startswith("POSTINGS THE USER CHOSE")
    assert "Experience with Kafka." in sent
    assert [s.label for s in data.job_sources] == [CHOSEN["label"]]
    assert data.requirements[0].frequency == 1


@pytest.mark.asyncio
async def test_extraction_of_other_sections_ignores_postings(monkeypatch):
    from agents.xbuddy.models import JobBuddyData

    model = RecordingExtraction(responses=['{"current_role": "Developer"}'])
    monkeypatch.setattr(extraction_mod, "get_chat_model", lambda config=None: model)

    await extraction_mod.extract_section_data(
        SectionID.BACKGROUND, [HumanMessage("developer")], "Developer", JobBuddyData(), CONFIG,
        postings=[CHOSEN],
    )  # fmt: skip

    assert "POSTINGS" not in model.prompts[0][1].content


def test_skill_gap_never_asks_for_a_link_and_offers_the_search():
    template = get_section_template(SectionID.SKILL_GAP)
    assert "URL" not in template.opening_question and "link" not in template.opening_question
    assert "find" in template.opening_question
    assert "Never ask for a link" in template.system_prompt_template


# --- the whole graph -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_picking_a_search_result_reads_it_and_keeps_it_for_the_section(
    servers, monkeypatch
):
    from agents.xbuddy.agent import graph

    servers(tracker=[httpx.Response(200, json=[GREENHOUSE_ROW])])
    _decision_stays(monkeypatch)
    model = use_reply_model(
        monkeypatch,
        read_request(GREENHOUSE),
        AIMessage("Braze - Senior Software Engineer I asks for '5+ years of Go or Java'."),
    )
    cfg = {"configurable": {"thread_id": "pick-1", "user_id": 1}}
    await _thread_in_skill_gap(graph, cfg)

    out = await graph.ainvoke({"messages": [HumanMessage("Use number 5, the Braze one")]}, cfg)

    assert isinstance(model.prompts[1][-1], ToolMessage)  # the reply saw the posting
    assert [p["url"] for p in out["fetched_postings"]] == [GREENHOUSE]
    assert [type(m) for m in out["messages"]] == [HumanMessage, AIMessage]
    assert out["tool_scratch"] == []

    # Next turn: no tool call, but the posting is still in the Skill Gap prompt.
    later = use_reply_model(monkeypatch, AIMessage("Kafka is your main gap."))
    await graph.ainvoke({"messages": [HumanMessage("What am I missing?")]}, cfg)
    assert "=== Braze - Senior Software Engineer I, Email" in later.prompts[0][0].content


"""PR 3 tests: generate_reply and generate_decision with fake models."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import XBuddyState, default_state
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes.generate_decision import (
    DECISION_TAG,
    apply_guardrails,
    generate_decision_node,
    missing_required_fields,
    parse_decision,
)
from agents.xbuddy.nodes.generate_reply import FALLBACK_REPLY, generate_reply_node
from agents.xbuddy.prompts import get_section_template
from agents.xbuddy.tools import build_context_packet

CONFIG = {"configurable": {"thread_id": "t", "user_id": 1}}
BACKGROUND_FIELDS = get_section_template(SectionID.BACKGROUND).required_fields


class RecordingModel(FakeListChatModel):
    """Fake model that remembers the prompt and config of each call."""

    calls: list = Field(default_factory=list)

    async def ainvoke(self, input, config=None, **kwargs):
        self.calls.append((input, config))
        return await super().ainvoke(input, config, **kwargs)


class BrokenModel(FakeListChatModel):
    async def ainvoke(self, input, config=None, **kwargs):
        raise TimeoutError("provider timed out")


def use_model(monkeypatch, module, model):
    monkeypatch.setattr(module, "get_chat_model", lambda config=None: model)
    return model


def base_state(**extra) -> dict:
    state = {"messages": [], "thread_id": "t", "user_id": 1, **default_state()}
    state["section_states"]["background"] = state["section_states"]["background"].model_copy(
        update={"status": SectionStatus.IN_PROGRESS}
    )
    state["context_packet"] = build_context_packet(SectionID.BACKGROUND, state["section_states"])
    state.update(extra)
    return state


# --- generate_reply -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reply_uses_section_prompt_and_short_memory(monkeypatch):
    model = use_model(monkeypatch, reply_mod, RecordingModel(responses=["How many years?"]))
    asked = AIMessage("What is your current job title?")
    user = HumanMessage("Backend developer")
    state = base_state(messages=[asked, user], short_memory=[asked])

    result = await generate_reply_node(state, CONFIG)

    prompt, _ = model.calls[0]
    assert isinstance(prompt[0], SystemMessage)
    assert "SECTION: Background" in prompt[0].content
    assert [m.content for m in prompt[1:]] == ["What is your current job title?", "Backend developer"]
    assert result["messages"][0].content == "How many years?"
    assert [m.content for m in result["short_memory"]][-2:] == ["Backend developer", "How many years?"]
    assert result["awaiting_user_input"] is True


@pytest.mark.asyncio
async def test_reply_provider_error_returns_fallback_and_records_error(monkeypatch):
    use_model(monkeypatch, reply_mod, BrokenModel(responses=["unused"]))
    result = await generate_reply_node(base_state(messages=[HumanMessage("hi")]), CONFIG)

    assert result["messages"][0].content == FALLBACK_REPLY
    assert result["error_count"] == 1
    assert "TimeoutError" in result["last_error"]


@pytest.mark.asyncio
async def test_reply_tokens_stream_through_langgraph(monkeypatch):
    model = GenericFakeChatModel(messages=iter([AIMessage("Great to meet you. What is your title?")]))
    use_model(monkeypatch, reply_mod, model)
    graph = StateGraph(XBuddyState)
    graph.add_node("generate_reply", generate_reply_node)
    graph.add_edge(START, "generate_reply")
    graph.add_edge("generate_reply", END)

    chunks = [
        chunk.content
        async for chunk, meta in graph.compile().astream(
            base_state(messages=[HumanMessage("hi")]), CONFIG, stream_mode="messages"
        )
        if meta.get("langgraph_node") == "generate_reply"
    ]
    assert len(chunks) > 1  # streamed token by token, not one final blob
    assert "".join(chunks) == "Great to meet you. What is your title?"


# --- parse_decision: valid and malformed output ----------------------------------------------

def test_parse_valid_decision():
    d = parse_decision(
        '{"router_directive": "next", "is_satisfied": true, "should_save_content": true,'
        ' "section_summary": "Backend dev, 3 years"}'
    )
    assert d.router_directive == "next"
    assert d.section_summary == "Backend dev, 3 years"


def test_parse_decision_in_code_fence_with_prose():
    d = parse_decision('Sure!\n```json\n{"router_directive": "STAY"}\n```')
    assert d.router_directive == "stay"


@pytest.mark.parametrize(
    "raw",
    [
        "I think we should move on.",                     # no JSON
        '{"router_directive": "next",}',                  # broken JSON
        '{"router_directive": "jump"}',                   # invalid directive
        '{"is_satisfied": true}',                         # missing directive
        '["next"]',                                       # not an object
        "",
    ],
)
def test_parse_malformed_decision_raises(raw):
    with pytest.raises(ValueError):
        parse_decision(raw)


# --- guardrails ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"router_directive": "next", "is_satisfied": True, "section_summary": "ok",
          "covered_fields": BACKGROUND_FIELDS}, "next"),
        ({"router_directive": "next", "is_satisfied": True, "section_summary": " ",
          "covered_fields": BACKGROUND_FIELDS}, "stay"),
        ({"router_directive": "next", "is_satisfied": None, "section_summary": "ok",
          "covered_fields": BACKGROUND_FIELDS}, "stay"),
        ({"router_directive": "modify:background"}, "stay"),        # current section
        ({"router_directive": "modify:salary"}, "stay"),            # unknown section
        ({"router_directive": "modify:target_role"}, "modify:target_role"),
    ],
)
def test_guardrails(fields, expected):
    decision = decision_mod.ChatAgentDecision(**fields)
    assert apply_guardrails(decision, SectionID.BACKGROUND).router_directive == expected


# --- completeness: `next` needs every required checklist item ---------------------------------

def confirmed(covered: list[str]) -> "decision_mod.ChatAgentDecision":
    return decision_mod.ChatAgentDecision(
        router_directive="next", is_satisfied=True, section_summary="ok", covered_fields=covered)


def test_next_refused_when_a_required_field_is_missing():
    covered = [f for f in BACKGROUND_FIELDS if f != "education"]
    decision = confirmed(covered)

    assert missing_required_fields(decision, SectionID.BACKGROUND) == ["education"]
    assert apply_guardrails(decision, SectionID.BACKGROUND).router_directive == "stay"


def test_next_refused_when_covered_fields_is_omitted():
    """Fails closed: a model that doesn't report coverage cannot advance."""
    decision = confirmed([])
    assert missing_required_fields(decision, SectionID.BACKGROUND) == BACKGROUND_FIELDS
    assert apply_guardrails(decision, SectionID.BACKGROUND).router_directive == "stay"


def test_fields_from_another_section_do_not_count():
    other = get_section_template(SectionID.TARGET_ROLE).required_fields
    assert apply_guardrails(confirmed(other), SectionID.BACKGROUND).router_directive == "stay"


@pytest.mark.parametrize("sid", list(SectionID))
def test_next_allowed_for_every_section_once_all_fields_are_covered(sid):
    fields = [f.upper() + " " for f in get_section_template(sid).required_fields]  # case/space
    decision = confirmed(fields)
    assert missing_required_fields(decision, sid) == []
    assert apply_guardrails(decision, sid).router_directive == "next"


@pytest.mark.asyncio
async def test_decision_node_stays_when_model_advances_with_missing_fields(monkeypatch):
    raw = json.dumps({"router_directive": "next", "is_satisfied": True,
                      "section_summary": "Backend dev", "covered_fields": ["current_role"]})
    use_model(monkeypatch, decision_mod, FakeListChatModel(responses=[raw]))
    result = await generate_decision_node(
        base_state(messages=exchange("Yes", "Great, moving on!")), CONFIG)

    assert result["router_directive"] == "stay"
    assert result["agent_output"].covered_fields == ["current_role"]
    assert "error_count" not in result  # a downgrade is a decision, not an error


# --- generate_decision node ------------------------------------------------------------------

def exchange(user_text: str, reply_text: str) -> list:
    return [
        AIMessage("Summary: backend dev, 3 years, Python. Does this look right?"),
        HumanMessage(user_text),
        AIMessage(reply_text),
    ]


@pytest.mark.asyncio
async def test_decision_valid_next(monkeypatch):
    raw = json.dumps({
        "router_directive": "next", "is_satisfied": True, "should_save_content": True,
        "section_summary": "Backend dev, 3 years, Python", "covered_fields": BACKGROUND_FIELDS,
    })
    model = use_model(monkeypatch, decision_mod, RecordingModel(responses=[raw]))
    state = base_state(messages=exchange("Yes, looks good", "Great! What titles next?"))

    result = await generate_decision_node(state, CONFIG)

    assert result["router_directive"] == "next"
    assert result["agent_output"].section_summary == "Backend dev, 3 years, Python"
    assert result["agent_output"].reply == "Great! What titles next?"
    assert "error_count" not in result
    _, config = model.calls[0]
    assert DECISION_TAG in config["tags"]  # hidden from /stream


@pytest.mark.asyncio
async def test_decision_valid_modify(monkeypatch):
    use_model(monkeypatch, decision_mod, FakeListChatModel(responses=[
        '{"router_directive": "modify:background", "is_satisfied": false}'
    ]))
    state = base_state(current_section=SectionID.TARGET_ROLE,
                       messages=exchange("Wait, I have 5 years not 3", "Let's fix that."))
    result = await generate_decision_node(state, CONFIG)
    assert result["router_directive"] == "modify:background"


@pytest.mark.asyncio
async def test_decision_malformed_output_stays_and_records_error(monkeypatch):
    use_model(monkeypatch, decision_mod, FakeListChatModel(responses=["Let's go to the next one!"]))
    state = base_state(messages=exchange("Yes", "Great!"), error_count=2)

    result = await generate_decision_node(state, CONFIG)

    assert result["router_directive"] == "stay"
    assert result["error_count"] == 3
    assert "no JSON" in result["last_error"]


@pytest.mark.asyncio
async def test_decision_premature_next_is_downgraded(monkeypatch):
    use_model(monkeypatch, decision_mod, FakeListChatModel(responses=[
        '{"router_directive": "next", "is_satisfied": true}'  # no summary collected
    ]))
    result = await generate_decision_node(base_state(messages=exchange("ok", "Next!")), CONFIG)
    assert result["router_directive"] == "stay"


@pytest.mark.asyncio
async def test_decision_without_reply_skips_model(monkeypatch):
    use_model(monkeypatch, decision_mod, BrokenModel(responses=["unused"]))
    result = await generate_decision_node(base_state(messages=[HumanMessage("hi")]), CONFIG)
    assert result["router_directive"] == "stay"
    assert "error_count" not in result

"""Skill Gap: gaps come from pasted postings, never from self-assessment."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import default_state
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.prompts import get_section_template
from agents.xbuddy.tools import build_context_packet

CONFIG = {"configurable": {"thread_id": "t", "user_id": 1}}
CONFIRM_SKILL_GAP = {
    "router_directive": "next", "is_satisfied": True, "should_save_content": True,
    "section_summary": "Gaps: Kafka. Action: Kafka course.",
    "covered_fields": ["job_sources", "requirements", "gaps", "learning_actions"],
}  # fmt: skip


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
    state["messages"] = [AIMessage("Here is the summary. Confirm?"), HumanMessage("yes"),
                         AIMessage("Thanks! Next: how do you plan to apply")]
    state.update(extra)
    return state


def decide(monkeypatch, decision: dict):
    monkeypatch.setattr(
        decision_mod, "get_chat_model",
        lambda config=None: FakeListChatModel(responses=[json.dumps(decision)]),
    )  # fmt: skip
    return decision_mod.generate_decision_node


@pytest.mark.asyncio
async def test_skill_gap_cannot_finish_without_a_pasted_posting(monkeypatch):
    node = decide(monkeypatch, CONFIRM_SKILL_GAP)
    self_assessed = [HumanMessage("I am weak at Kubernetes and AWS."), AIMessage("Noted.")]
    result = await node(skill_gap_state(short_memory=self_assessed), CONFIG)
    assert result["router_directive"] == "stay"


@pytest.mark.asyncio
async def test_skill_gap_can_finish_with_a_pasted_posting(monkeypatch):
    node = decide(monkeypatch, CONFIRM_SKILL_GAP)
    pasted = HumanMessage("Maple - Backend Developer\n" + "Requirements: Python, Django. " * 20)
    result = await node(skill_gap_state(short_memory=[pasted]), CONFIG)
    assert result["router_directive"] == "next"


@pytest.mark.asyncio
async def test_other_sections_do_not_need_a_posting(monkeypatch):
    decision = {**CONFIRM_SKILL_GAP,
                "covered_fields": get_section_template(SectionID.BACKGROUND).required_fields}
    node = decide(monkeypatch, decision)
    state = skill_gap_state()
    state["current_section"] = SectionID.BACKGROUND
    result = await node(state, CONFIG)
    assert result["router_directive"] == "next"


def test_skill_gap_asks_for_pasted_text_never_a_link_or_a_search():
    template = get_section_template(SectionID.SKILL_GAP)
    question = template.opening_question
    assert "Paste" in question and "link" not in question and "find" not in question
    prompt = " ".join(template.system_prompt_template.split())
    assert "Never ask for a link" in prompt
    assert "search" not in prompt.lower()


def test_skill_gap_prompt_forbids_asking_for_self_assessed_gaps():
    prompt = get_section_template(SectionID.SKILL_GAP).system_prompt_template
    assert "Never ask the user which skills they lack" in prompt
    assert "Until at least one posting has been pasted" in prompt


# --- live run Oct 17: Skill Gap stuck while the chat ran ahead -------------------------------

def test_reply_prompt_states_real_progress_and_forbids_a_fake_roadmap():
    state = skill_gap_state()
    prompt = state["context_packet"].system_prompt
    assert "- Skill Gap: in progress <- CURRENT" in prompt
    assert "- Application Strategy: not started" in prompt
    assert "Never write the roadmap yourself" in prompt


@pytest.mark.asyncio
async def test_the_decision_keeps_seeing_the_posting_in_a_long_section(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel as Fake

    posting = HumanMessage("Fintech company - GenAI Engineer\n" + "Requirements: RAG, LangChain. " * 100)
    chatter = []
    for i in range(9):
        chatter += [AIMessage(f"question {i}?"), HumanMessage(f"answer {i}")]
    memory = [AIMessage("Paste a posting"), posting, *chatter]

    seen = []

    class Recording(Fake):
        async def ainvoke(self, input, config=None, **kwargs):
            seen.append(input)
            return await super().ainvoke(input, config, **kwargs)

    monkeypatch.setattr(decision_mod, "get_chat_model",
                        lambda config=None: Recording(responses=[json.dumps(CONFIRM_SKILL_GAP)]))
    state = skill_gap_state(short_memory=memory, messages=[*memory, AIMessage("Thanks!")])
    result = await decision_mod.generate_decision_node(state, CONFIG)

    transcript = seen[0][1].content
    assert "Fintech company - GenAI Engineer" in transcript  # still in view after 20 messages
    assert "characters in total]" in transcript  # but cut, to save tokens
    assert len(transcript) < 4000
    assert result["router_directive"] == "next"


def test_one_posting_is_enough():
    """Live run: JobBuddy kept asking for a second posting after getting one."""
    template = get_section_template(SectionID.SKILL_GAP)
    prompt = " ".join(template.system_prompt_template.split())
    assert "One posting is enough" in prompt and "Never ask for more postings" in prompt
    assert "two or three" not in template.opening_question

"""End-to-end regressions for PR 11, through the real graph (fake models only).

Each test reproduces a bug seen in a live run and checks the user-visible
outcome: the section the sidebar shows (service.section_progress), the reply
the user gets, and whether a roadmap exists.
"""

import json
import re

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from test_roadmap import GOOD_ROADMAP, FakeStore, completed_state

from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import implementation as impl
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.nodes.generate_reply import FALLBACK_REPLY
from service.service import section_progress

# Which Background checklist item each phrase in a user message provides.
BACKGROUND_EVIDENCE = {
    "current_role": "backend developer",
    "role_history": "qa intern",
    "years_experience": "3 years",
    "skills": "python",
    "education": "msc",
}


class TranscriptReadingDecision:
    """A decision model that, like the real one, can only report the checklist
    items it can SEE in the transcript it is given, and says 'next' when the
    user confirms. It reproduces the PR 3 bug: shown only the last exchange, it
    cannot see facts given earlier, so 'next' is downgraded and the section sticks."""

    def __init__(self):
        self.transcripts = []

    async def ainvoke(self, messages, config=None):
        transcript = messages[-1].content
        self.transcripts.append(transcript)
        user_lines = " ".join(
            line for line in transcript.splitlines() if line.startswith(("USER:", "SAVED DRAFT"))
        ).lower() + " " + transcript.lower().split("saved draft of this section:", 1)[-1][:400]
        covered = [field for field, phrase in BACKGROUND_EVIDENCE.items() if phrase in user_lines]
        confirmed = re.search(r"USER: (yes|looks good)", transcript, re.IGNORECASE) is not None
        return AIMessage(json.dumps({
            "router_directive": "next" if confirmed else "stay",
            "is_satisfied": confirmed or None,
            "should_save_content": False,
            "section_summary": "Background so far" if confirmed else None,
            "covered_fields": covered,
        }))


def use(monkeypatch, module, model):
    monkeypatch.setattr(module, "get_chat_model", lambda config=None: model)
    return model


@pytest.fixture(autouse=True)
def no_stores(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(mu, "get_section_store", lambda: store)
    monkeypatch.setattr(impl, "get_roadmap_store", lambda: store)
    return store


def sidebar(state) -> dict:
    progress = section_progress(state)
    return {s["id"]: s["status"] for s in progress["sections"]} | {"current": progress["id"]}


async def background_turns(graph, cfg, monkeypatch, answers):
    """Answer Background across several turns, then confirm. Returns the final state."""
    use(monkeypatch, decision_mod, decision := TranscriptReadingDecision())
    use(monkeypatch, extraction_mod, FakeListChatModel(responses=[json.dumps({
        "current_role": "Backend developer", "role_history": ["QA intern"],
        "years_experience": 3, "skills": ["Python"], "education": "MSc"})] * 3))
    await graph.ainvoke({"messages": []}, cfg)
    state = None
    for answer in [*answers, "Yes, looks good"]:
        use(monkeypatch, reply_mod, FakeListChatModel(responses=["Noted. Next question?"]))
        state = await graph.ainvoke({"messages": [HumanMessage(answer)]}, cfg)
    return state, decision


# --- routing (the "sidebar stuck on Background" bug) -----------------------------------------

@pytest.mark.asyncio
async def test_a_section_answered_over_several_turns_advances_on_confirmation(monkeypatch):
    """Live bug (PR 3): facts given in earlier turns were invisible to the decision, so
    the confirmed section stayed on Background. Fails on the old last-exchange transcript."""
    from agents.xbuddy.agent import graph

    cfg = {"configurable": {"thread_id": "e2e-route-1", "user_id": 1}}
    state, decision = await background_turns(graph, cfg, monkeypatch, [
        "Backend developer",
        "Before that, QA intern",
        "3 years",
        "Python and Django",
        "MSc Computer Science",
    ])

    assert sidebar(state)["background"] == "done"
    assert sidebar(state)["current"] == "target_role"
    assert "Backend developer" in decision.transcripts[-1]  # turn 1 still visible at the end


@pytest.mark.asyncio
async def test_an_incomplete_section_stays_put_even_when_the_user_says_yes(monkeypatch):
    from agents.xbuddy.agent import graph

    cfg = {"configurable": {"thread_id": "e2e-route-2", "user_id": 1}}
    state, _ = await background_turns(graph, cfg, monkeypatch, [
        "Backend developer, before that QA intern, 3 years, Python",  # no education
    ])

    assert sidebar(state)["background"] == "in_progress"
    assert sidebar(state)["current"] == "background"
    assert state["section_states"]["background"].status == SectionStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_a_detailed_answer_completes_the_section_in_one_turn(monkeypatch):
    from agents.xbuddy.agent import graph

    cfg = {"configurable": {"thread_id": "e2e-route-3", "user_id": 1}}
    state, _ = await background_turns(graph, cfg, monkeypatch, [
        "Backend developer for 3 years, before that QA intern. Python. MSc Computer Science.",
    ])

    assert sidebar(state)["current"] == "target_role"
    assert state["user_data"].profile.education == "MSc"  # every supplied field was saved


# --- empty replies -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_empty_model_reply_never_reaches_the_user(monkeypatch):
    """Live bug: the model returned an empty message and the browser showed 'reply stopped'."""
    from agents.xbuddy.agent import graph

    cfg = {"configurable": {"thread_id": "e2e-empty", "user_id": 1}}
    await graph.ainvoke({"messages": []}, cfg)
    use(monkeypatch, decision_mod, FakeListChatModel(responses=[json.dumps({"router_directive": "stay"})]))
    use(monkeypatch, reply_mod, FakeListChatModel(responses=["", "What's your current role?"]))

    state = await graph.ainvoke({"messages": [HumanMessage("hi")]}, cfg)

    assert state["messages"][-1].content == "What's your current role?"
    assert all(m.content for m in state["messages"] if isinstance(m, AIMessage))

    use(monkeypatch, reply_mod, FakeListChatModel(responses=["", " "]))
    use(monkeypatch, decision_mod, FakeListChatModel(responses=[json.dumps({"router_directive": "stay"})]))
    state = await graph.ainvoke({"messages": [HumanMessage("hello?")]}, cfg)
    assert state["messages"][-1].content == FALLBACK_REPLY


# --- the completion gate (no roadmap while a question is open) ------------------------------

async def thread_in_last_section(graph, cfg):
    state = completed_state(should_generate_final_output=False, router_directive="stay")
    state["section_states"]["interview_prep"] = state["section_states"]["interview_prep"].model_copy(
        update={"status": SectionStatus.IN_PROGRESS, "confirmed_summary": None})
    state["messages"] = [AIMessage("Summary: technical + system design interviews. Right?")]
    await graph.aupdate_state(cfg, state)


CONFIRM_LAST = json.dumps({
    "router_directive": "next", "is_satisfied": True, "should_save_content": True,
    "section_summary": "Technical + system design",
    "covered_fields": ["interview_types", "weak_areas", "practice_plan", "stories"]})


@pytest.mark.asyncio
async def test_no_roadmap_while_the_reply_asks_a_question(monkeypatch, no_stores):
    """Live bug: the roadmap appeared in the same turn JobBuddy asked a question."""
    from agents.xbuddy.agent import graph

    cfg = {"configurable": {"thread_id": "e2e-gate", "user_id": 1}}
    await thread_in_last_section(graph, cfg)
    use(monkeypatch, reply_mod, FakeListChatModel(responses=["Which formats do you expect?"]))
    use(monkeypatch, decision_mod, FakeListChatModel(responses=[CONFIRM_LAST]))
    use(monkeypatch, impl, FakeListChatModel(responses=[GOOD_ROADMAP]))

    asked = await graph.ainvoke({"messages": [HumanMessage("I'm weak at system design")]}, cfg)

    assert asked["roadmap"] is None
    assert asked["finished"] is False
    assert sidebar(asked)["interview_prep"] == "in_progress"
    assert no_stores.roadmaps == []

    # The user answers; the closing reply has no question, so the roadmap is written.
    use(monkeypatch, reply_mod, FakeListChatModel(responses=["Thanks! Putting your roadmap together."]))
    use(monkeypatch, decision_mod, FakeListChatModel(responses=[CONFIRM_LAST]))
    done = await graph.ainvoke({"messages": [HumanMessage("Technical and system design. Yes.")]}, cfg)

    assert done["roadmap"] == GOOD_ROADMAP.strip()
    assert done["finished"] is True
    assert section_progress(done)["roadmap_ready"] is True


# --- the reply can't run ahead of the graph --------------------------------------------------

@pytest.mark.asyncio
async def test_every_reply_prompt_carries_the_real_section_statuses(monkeypatch):
    """Live bug: Skill Gap was stuck but the reply walked on and wrote a fake roadmap."""
    from agents.xbuddy.agent import graph

    class Recording(FakeListChatModel):
        seen: list = []

        async def ainvoke(self, input, config=None, **kwargs):
            Recording.seen.append(input[0].content)
            return await super().ainvoke(input, config, **kwargs)

    Recording.seen = []
    cfg = {"configurable": {"thread_id": "e2e-progress", "user_id": 1}}
    await graph.ainvoke({"messages": []}, cfg)
    use(monkeypatch, decision_mod, FakeListChatModel(responses=[json.dumps({"router_directive": "stay"})]))
    use(monkeypatch, reply_mod, Recording(responses=["What's your current role?"]))

    await graph.ainvoke({"messages": [HumanMessage("hi")]}, cfg)

    prompt = Recording.seen[-1]
    assert "- Background: in progress <- CURRENT" in prompt
    assert "- Skill Gap: not started" in prompt
    assert "Never write the roadmap yourself" in prompt


# --- a refused confirmation steers the next reply back (live site, Oct 19) -------------------

@pytest.mark.asyncio
async def test_a_refused_confirmation_steers_the_next_reply_back_to_the_section(monkeypatch):
    """Live bug: Skill Gap's confirmation was refused, the chat ran on through the last
    two sections and ended with 'everything's set', and no roadmap was ever written."""
    from agents.xbuddy.agent import graph

    class Recording(FakeListChatModel):
        seen: list = []

        async def ainvoke(self, input, config=None, **kwargs):
            Recording.seen.append(input[0].content)
            return await super().ainvoke(input, config, **kwargs)

    cfg = {"configurable": {"thread_id": "e2e-steer", "user_id": 1}}
    state = completed_state(should_generate_final_output=False, router_directive="stay")
    for sid in ("skill_gap", "application_strategy", "interview_prep"):
        state["section_states"][sid] = state["section_states"][sid].model_copy(
            update={"status": SectionStatus.PENDING, "confirmed_summary": None})
    state["section_states"]["skill_gap"] = state["section_states"]["skill_gap"].model_copy(
        update={"status": SectionStatus.IN_PROGRESS})
    state["current_section"] = SectionID.SKILL_GAP
    posting = HumanMessage("Fintech - GenAI Engineer\n" + "Requirements: RAG, LangChain, Kubernetes. " * 20)
    state["messages"] = [posting, AIMessage("Gaps: Kubernetes. Learning action: a course. Confirm?")]
    state["short_memory"] = list(state["messages"])
    await graph.aupdate_state(cfg, state)

    # The user confirms, but the decision misses a checklist item: refused, reason stored.
    use(monkeypatch, reply_mod, FakeListChatModel(responses=["Thanks! How are you applying today?"]))
    use(monkeypatch, decision_mod, FakeListChatModel(responses=[json.dumps({
        "router_directive": "next", "is_satisfied": True, "section_summary": "Gaps: Kubernetes",
        "covered_fields": ["job_sources", "requirements", "gaps"]})]))
    refused = await graph.ainvoke({"messages": [HumanMessage("yes")]}, cfg)

    assert sidebar(refused)["current"] == "skill_gap"
    assert "learning_actions" in refused["blocked_next"]

    # Next turn: the reply is told to go back to Skill Gap before anything else.
    Recording.seen = []
    use(monkeypatch, reply_mod, Recording(responses=["One more thing for Skill Gap: which learning action?"]))
    use(monkeypatch, decision_mod, FakeListChatModel(responses=[json.dumps({"router_directive": "stay"})]))
    steered = await graph.ainvoke({"messages": [HumanMessage("LinkedIn and career pages")]}, cfg)

    prompt = Recording.seen[-1]
    assert "IMPORTANT, DO THIS FIRST" in prompt and "learning_actions" in prompt
    assert steered["blocked_next"] is None  # the reason is used once, then cleared

    # Once the section is complete it advances as normal.
    use(monkeypatch, reply_mod, FakeListChatModel(responses=["Thanks! How are you applying today?"]))
    use(monkeypatch, decision_mod, FakeListChatModel(responses=[json.dumps({
        "router_directive": "next", "is_satisfied": True, "section_summary": "Gaps: Kubernetes; course",
        "covered_fields": ["job_sources", "requirements", "gaps", "learning_actions"]})]))
    use(monkeypatch, extraction_mod, FakeListChatModel(responses=['{"job_sources": [], "requirements": []}'] * 3))
    done = await graph.ainvoke({"messages": [HumanMessage("A Kubernetes course. Yes.")]}, cfg)

    assert sidebar(done)["current"] == "application_strategy"

"""PR 5 tests: implementation node, roadmap quality criteria, and grounding."""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from agents.xbuddy.enums import RequirementPriority, SectionID, SectionStatus, WorkType
from agents.xbuddy.models import (
    ChatAgentOutput,
    JobBuddyData,
    JobRequirement,
    JobSource,
    RequirementEvidence,
    SectionState,
    TargetRole,
    UserProfile,
    default_state,
)
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import implementation as impl
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.nodes.generate_reply import generate_reply_node
from agents.xbuddy.nodes.implementation import ROADMAP_TAG, implementation_node
from agents.xbuddy.nodes.memory_updater import memory_updater_node
from agents.xbuddy.prompts import get_section_template
from agents.xbuddy.roadmap import (
    build_roadmap_input,
    fallback_roadmap,
    find_contradictions,
    requirements_table,
    validate_roadmap,
)

CONFIG = {"configurable": {"thread_id": "t", "user_id": 1}}

GOOD_ROADMAP = """# Your Job Search Roadmap

## Where you are now
Backend developer with 3 years of Python.

## What you're aiming for
Senior backend roles in Toronto, hybrid.

## Skill gaps to close
- Kubernetes: 2 of 2 postings (Shopify - Backend, Wealthsimple - SWE)
- Docker: 1 of 2 postings (Shopify - Backend)

## Learning plan
Finish a Kubernetes course and deploy one project.

## Application strategy
Ten applications a week, two referrals.

## Interview preparation
Two mock interviews a week.

## 8-week plan
| Week | Focus | Concrete targets |
| --- | --- | --- |
| 1 | Resume | Rewrite resume |
| 2 | Kubernetes | Finish module 1 |
| 3 | Applications | 10 applications |
| 4 | Kubernetes | Deploy a project |
| 5 | Applications | 10 applications |
| 6 | Interviews | 2 mocks |
| 7 | Applications | 10 applications |
| 8 | Review | Adjust the plan |

## This week's checklist
- [ ] Rewrite resume summary
- [ ] Update LinkedIn
- [ ] Enrol in the Kubernetes course
- [ ] Shortlist 10 roles
- [ ] Message two contacts
"""


class FakeStore:
    def __init__(self, fail=False):
        self.roadmaps, self.sections, self.fail = [], [], fail

    async def save_roadmap(self, user_id, thread_id, roadmap):
        if self.fail:
            raise RuntimeError("supabase down")
        self.roadmaps.append(roadmap)

    async def save_section(self, user_id, thread_id, state):
        self.sections.append(state)


class Scripted(FakeListChatModel):
    """Returns the given responses in order and records each prompt and config."""

    calls: list = Field(default_factory=list)

    async def ainvoke(self, input, config=None, **kwargs):
        self.calls.append((input, config))
        return await super().ainvoke(input, config, **kwargs)


class BrokenModel(FakeListChatModel):
    async def ainvoke(self, input, config=None, **kwargs):
        raise TimeoutError("provider timed out")


def fake(monkeypatch, module, model):
    monkeypatch.setattr(module, "get_chat_model", lambda config=None: model)
    return model


@pytest.fixture
def store(monkeypatch):
    s = FakeStore()
    monkeypatch.setattr(impl, "get_roadmap_store", lambda: s)
    monkeypatch.setattr(mu, "get_section_store", lambda: s)
    return s


def req(name, sources, priority=RequirementPriority.MUST_HAVE):
    return JobRequirement(name=name, priority=priority, evidence=[
        RequirementEvidence(source_label=src, quote=f"{name} required") for src in sources
    ])


def user_data() -> JobBuddyData:
    return JobBuddyData(
        job_sources=[JobSource(label="Shopify - Backend"), JobSource(label="Wealthsimple - SWE")],
        requirements=[req("Docker", ["Shopify - Backend"], RequirementPriority.NICE_TO_HAVE),
                      req("Kubernetes", ["Shopify - Backend", "Wealthsimple - SWE"])],
    )


def completed_state(**extra) -> dict:
    state = {"messages": [], "thread_id": "t", "user_id": 1, **default_state()}
    for sid in SectionID:
        state["section_states"][sid.value] = SectionState(
            section_id=sid, status=SectionStatus.DONE, confirmed_summary=f"{sid.value} confirmed")
    state["current_section"] = SectionID.INTERVIEW_PREP
    state["user_data"] = user_data()
    state["should_generate_final_output"] = True
    state.update(extra)
    return state


# --- inputs ----------------------------------------------------------------------------------

def test_requirements_ranked_by_demand_with_sources():
    lines = requirements_table(user_data()).splitlines()
    assert lines[0].startswith("- Kubernetes (must-have; 2 of 2 postings:")
    assert "Shopify - Backend, Wealthsimple - SWE" in lines[0]
    assert lines[1].startswith("- Docker (nice-to-have; 1 of 2 postings: Shopify - Backend)")


def test_roadmap_input_uses_only_confirmed_sections():
    state = completed_state()
    state["section_states"]["interview_prep"] = SectionState(
        section_id=SectionID.INTERVIEW_PREP, status=SectionStatus.IN_PROGRESS,
        confirmed_summary="stale")
    text = build_roadmap_input(state["section_states"], state["user_data"])
    assert "background confirmed" in text
    assert "stale" not in text
    assert "Kubernetes" in text


# --- quality criteria ------------------------------------------------------------------------

def test_good_roadmap_passes_every_check():
    assert validate_roadmap(GOOD_ROADMAP, user_data()) == []


def test_curly_apostrophes_in_headings_are_accepted():
    curly = GOOD_ROADMAP.replace("What you're", "What you’re").replace(
        "This week's", "This week’s")
    assert validate_roadmap(curly, user_data()) == []


@pytest.mark.parametrize(
    "broken, expected",
    [
        (GOOD_ROADMAP.replace("## Learning plan", "## Study"), "missing section: ## Learning plan"),
        (GOOD_ROADMAP.replace("| 8 | Review | Adjust the plan |\n", ""), "8-week plan has 7 week"),
        (GOOD_ROADMAP.replace("- [ ] Message two contacts\n", "").replace(
            "- [ ] Shortlist 10 roles\n", ""), "checklist has 3 items"),
        (GOOD_ROADMAP.replace("2 of 2 postings", "5 of 6 postings"), "ungrounded claim"),
        (GOOD_ROADMAP.replace("1 of 2 postings", "3 of 2 postings"), "ungrounded claim"),
        (GOOD_ROADMAP.replace("Kubernetes", "container orchestration"),
         "skill gap not mentioned: Kubernetes"),
        (GOOD_ROADMAP.replace("Rewrite resume |", "[TBD] |"), "placeholder text"),
        (GOOD_ROADMAP.replace("Senior backend roles", "Roles paying $150k"), "salary or money"),
        ("", "roadmap is empty"),
    ],
)
def test_each_quality_problem_is_detected(broken, expected):
    problems = validate_roadmap(broken, user_data())
    assert any(expected in p for p in problems), problems


# --- real model output and typography ----------------------------------------------------------

REAL_ROADMAP = (Path(__file__).parent / "fixtures" / "real_model_roadmap.md").read_text("utf-8")


def real_data() -> JobBuddyData:
    sources = ["Shopify - Backend Developer", "Wealthsimple - Software Engineer"]
    return JobBuddyData(
        profile=UserProfile(current_role="Backend Developer", years_experience=3),
        target_role=TargetRole(titles=["Senior Backend Developer"], work_type=WorkType.HYBRID),
        job_sources=[JobSource(label=s) for s in sources],
        requirements=[req("Kubernetes", sources),
                      req("Docker", sources[:1], RequirementPriority.NICE_TO_HAVE)],
    )


def test_real_model_roadmap_passes_every_check():
    """A roadmap openai/gpt-oss-120b actually produced. It writes "8-week" with a
    non-breaking hyphen (U+2011), uses en dashes and curly apostrophes, and bolds the
    week numbers. None of that is a quality problem."""
    assert "8\u2011week plan" in REAL_ROADMAP
    assert validate_roadmap(REAL_ROADMAP, real_data()) == []


@pytest.mark.parametrize(
    "heading",
    ["## 8\u2011week plan", "## 8\u2013week plan", "## **8-Week Plan**", "## 8-week plan:",
     "##  8-week   plan  "],
)
def test_heading_typography_variants_are_accepted(heading):
    variant = GOOD_ROADMAP.replace("## 8-week plan", heading)
    assert validate_roadmap(variant, user_data()) == []


def test_checkbox_styles_are_accepted():
    starred = GOOD_ROADMAP.replace("- [ ]", "* [ ]").replace("* [ ] Update LinkedIn", "- [x] Update LinkedIn")
    assert validate_roadmap(starred, user_data()) == []


def test_typography_does_not_hide_real_problems():
    broken = REAL_ROADMAP.replace("2 of 2 postings", "7 of 9 postings").replace(
        "3 years of experience", "12 years of experience")
    problems = validate_roadmap(broken, real_data())
    assert any("ungrounded claim '7 of 9 postings'" in p for p in problems)
    assert any("says '12 years' of experience" in p for p in problems)


def test_one_line_naming_both_priorities_is_not_a_contradiction():
    both = GOOD_ROADMAP.replace(
        "- Kubernetes: 2 of 2 postings (Shopify - Backend, Wealthsimple - SWE)\n"
        "- Docker: 1 of 2 postings (Shopify - Backend)",
        "- Kubernetes is a must-have (2 of 2 postings); Docker is a nice-to-have (1 of 2 postings)")
    assert find_contradictions(both, confirmed_data()) == []


# --- contradictions with confirmed data --------------------------------------------------------

def confirmed_data() -> JobBuddyData:
    data = user_data()
    return data.model_copy(update={
        "profile": UserProfile(current_role="Backend Developer", years_experience=3),
        "target_role": TargetRole(titles=["Senior Backend Developer"], work_type=WorkType.HYBRID),
    })


def test_consistent_roadmap_has_no_contradictions():
    assert find_contradictions(GOOD_ROADMAP, confirmed_data()) == []
    assert validate_roadmap(GOOD_ROADMAP, confirmed_data()) == []


@pytest.mark.parametrize(
    "broken, expected",
    [
        (GOOD_ROADMAP.replace("3 years of Python", "7 years of Python"),
         "says '7 years' of experience, the user confirmed 3"),
        (GOOD_ROADMAP.replace("Toronto, hybrid", "Toronto, fully remote"),
         "targets remote work, the user confirmed hybrid"),
        (GOOD_ROADMAP.replace("- Kubernetes: 2 of 2", "- Kubernetes (optional): 2 of 2"),
         "'Kubernetes' is must-have in the postings"),
        (GOOD_ROADMAP.replace("- Docker: 1 of 2", "- Docker (must-have): 1 of 2"),
         "'Docker' is nice-to-have in the postings"),
    ],
)
def test_each_contradiction_is_detected(broken, expected):
    problems = validate_roadmap(broken, confirmed_data())
    assert any(expected in p for p in problems), problems


def test_mentioning_the_confirmed_arrangement_alongside_another_is_allowed():
    mixed = GOOD_ROADMAP.replace("Toronto, hybrid", "Toronto, hybrid (two remote days)")
    assert find_contradictions(mixed, confirmed_data()) == []


def test_empty_section_is_reported():
    empty = GOOD_ROADMAP.replace("Finish a Kubernetes course and deploy one project.", "")
    assert "empty section: ## Learning plan" in validate_roadmap(empty, user_data())


@pytest.mark.asyncio
async def test_contradicting_roadmap_is_rewritten_with_that_feedback(monkeypatch, store):
    contradicting = GOOD_ROADMAP.replace("3 years of Python", "10 years of Python")
    model = fake(monkeypatch, impl, Scripted(responses=[contradicting, GOOD_ROADMAP]))
    result = await implementation_node(completed_state(user_data=confirmed_data()), CONFIG)

    assert result["roadmap"] == GOOD_ROADMAP.strip()
    assert "the user confirmed 3" in model.calls[1][0][-1].content


def test_fallback_roadmap_always_passes_the_checks():
    state = completed_state()
    assert validate_roadmap(
        fallback_roadmap(state["section_states"], state["user_data"]), state["user_data"]) == []
    empty = default_state()
    assert validate_roadmap(
        fallback_roadmap(empty["section_states"], JobBuddyData()), JobBuddyData()) == []


# --- implementation node ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generates_saves_and_delivers_a_validated_roadmap(monkeypatch, store):
    model = fake(monkeypatch, impl, Scripted(responses=[GOOD_ROADMAP]))
    result = await implementation_node(completed_state(), CONFIG)

    assert result["roadmap"] == GOOD_ROADMAP.strip()
    assert result["messages"][0].content == GOOD_ROADMAP.strip()
    assert result["finished"] is True
    assert result["should_generate_final_output"] is False
    assert result["router_directive"] == "stay"
    assert store.roadmaps == [GOOD_ROADMAP.strip()]
    assert "last_error" not in result
    assert len(model.calls) == 1
    assert ROADMAP_TAG in model.calls[0][1]["tags"]  # drafts are never streamed


@pytest.mark.asyncio
async def test_failed_roadmap_gets_one_rewrite_with_the_failed_checks(monkeypatch, store):
    ungrounded = GOOD_ROADMAP.replace("2 of 2 postings", "9 of 12 postings")
    model = fake(monkeypatch, impl, Scripted(responses=[ungrounded, GOOD_ROADMAP]))
    result = await implementation_node(completed_state(), CONFIG)

    assert result["roadmap"] == GOOD_ROADMAP.strip()
    assert "last_error" not in result
    assert len(model.calls) == 2
    feedback = model.calls[1][0][-1].content
    assert "ungrounded claim '9 of 12 postings'" in feedback


@pytest.mark.asyncio
async def test_roadmap_that_keeps_failing_is_replaced_by_the_fallback(monkeypatch, store):
    bad = "# Your Job Search Roadmap\n\nYou should learn Rust. 9 of 12 postings want it."
    model = fake(monkeypatch, impl, Scripted(responses=[bad]))
    state = completed_state()
    result = await implementation_node(state, CONFIG)

    assert len(model.calls) == 2  # original + one rewrite, then stop
    assert result["roadmap"] == fallback_roadmap(state["section_states"], state["user_data"])
    assert validate_roadmap(result["roadmap"], state["user_data"]) == []
    assert "Rust" not in result["roadmap"]  # the ungrounded draft is never delivered
    assert "ungrounded claim" in result["last_error"]
    assert result["finished"] is True


@pytest.mark.parametrize("model", [BrokenModel(responses=["x"]), FakeListChatModel(responses=["  "])])
@pytest.mark.asyncio
async def test_model_failure_falls_back_to_a_valid_roadmap(monkeypatch, store, model):
    fake(monkeypatch, impl, model)
    state = completed_state()
    result = await implementation_node(state, CONFIG)

    assert result["roadmap"] == fallback_roadmap(state["section_states"], state["user_data"])
    assert "background confirmed" in result["roadmap"]
    assert "Kubernetes (must-have; 2 of 2" in result["roadmap"]
    assert result["last_error"].startswith("implementation:")
    assert result["finished"] is True


@pytest.mark.asyncio
async def test_roadmap_save_failure_is_recorded(monkeypatch):
    monkeypatch.setattr(impl, "get_roadmap_store", lambda: FakeStore(fail=True))
    fake(monkeypatch, impl, FakeListChatModel(responses=[GOOD_ROADMAP]))
    result = await implementation_node(completed_state(), CONFIG)

    assert result["roadmap"] == GOOD_ROADMAP.strip()
    assert "persist[roadmap]: supabase down" in result["last_error"]


# --- after the roadmap -----------------------------------------------------------------------

def confirm(state, summary):
    state["agent_output"] = ChatAgentOutput(
        reply="ok", router_directive="next", is_satisfied=True,
        should_save_content=True, section_summary=summary)
    return state


@pytest.mark.asyncio
async def test_thanks_after_roadmap_does_not_regenerate(store):
    state = confirm(completed_state(roadmap=GOOD_ROADMAP, should_generate_final_output=False),
                    "interview_prep confirmed")
    result = await memory_updater_node(state, CONFIG)
    assert "should_generate_final_output" not in result


@pytest.mark.asyncio
async def test_correction_after_roadmap_regenerates(store):
    state = completed_state(roadmap=GOOD_ROADMAP, should_generate_final_output=False,
                            current_section=SectionID.APPLICATION_STRATEGY)
    state["section_states"]["application_strategy"] = state["section_states"][
        "application_strategy"].model_copy(update={"status": SectionStatus.IN_PROGRESS})
    result = await memory_updater_node(confirm(state, "Now 15 applications a week"), CONFIG)
    assert result["should_generate_final_output"] is True


@pytest.mark.asyncio
async def test_follow_up_reply_can_see_the_roadmap(monkeypatch):
    model = fake(monkeypatch, reply_mod, Scripted(responses=["Week 1 is about your resume."]))
    state = completed_state(roadmap=GOOD_ROADMAP, messages=[HumanMessage("What is week 1?")])
    await generate_reply_node(state, CONFIG)

    system = model.calls[0][0][0].content
    assert "ROADMAP ALREADY DELIVERED" in system
    assert GOOD_ROADMAP in system


# --- end to end ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_graph_last_confirmation_produces_roadmap(monkeypatch, store):
    from agents.xbuddy.agent import graph

    cfg = {"configurable": {"thread_id": "flow-roadmap", "user_id": 1}}
    state = completed_state(should_generate_final_output=False)
    state["section_states"]["interview_prep"] = state["section_states"][
        "interview_prep"].model_copy(update={"status": SectionStatus.IN_PROGRESS,
                                             "confirmed_summary": None})
    state["messages"] = [AIMessage("Summary: behavioral + system design. Right?")]
    await graph.aupdate_state(cfg, state)

    fake(monkeypatch, reply_mod, FakeListChatModel(
        responses=["Thanks! I'm putting together your roadmap now."]))
    fake(monkeypatch, decision_mod, FakeListChatModel(responses=[json.dumps({
        "router_directive": "next", "is_satisfied": True, "should_save_content": True,
        "section_summary": "Behavioral + system design",
        "covered_fields": get_section_template(SectionID.INTERVIEW_PREP).required_fields})]))
    fake(monkeypatch, impl, FakeListChatModel(responses=[GOOD_ROADMAP]))

    final = await graph.ainvoke({"messages": [HumanMessage("Yes, that's right")]}, cfg)

    assert final["roadmap"] == GOOD_ROADMAP.strip()
    assert final["messages"][-1].content == GOOD_ROADMAP.strip()
    assert final["finished"] is True
    assert all(s.status == SectionStatus.DONE for s in final["section_states"].values())
    assert store.roadmaps == [GOOD_ROADMAP.strip()]

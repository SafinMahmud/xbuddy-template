"""PR 5 evaluation: one complete five-section conversation, scored end to end.

Drives the real graph (initialize, router, reply, decision, memory updater,
implementation) with scripted models, two turns per section, then checks the
final roadmap against the quality criteria and against the conversation's data.
"""

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field

from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.graph.builder import build_xbuddy_graph
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import implementation as impl
from agents.xbuddy.nodes import memory_updater as mu
from agents.xbuddy.prompts import get_section_template
from agents.xbuddy.roadmap import REQUIRED_HEADINGS, find_contradictions, validate_roadmap

# What the "user" confirms in each section, and the structured data extracted from it.
CONVERSATION = {
    SectionID.BACKGROUND: (
        "Backend developer, 3 years, Python and Django, MSc Computer Science",
        {"current_role": "Backend Developer", "role_history": ["Junior Web Developer"],
         "years_experience": 3, "skills": ["Python", "Django"], "education": "MSc CS"},
    ),
    SectionID.TARGET_ROLE: (
        "Senior backend roles in Toronto, hybrid, priority is growth",
        {"titles": ["Senior Backend Developer"], "location_region": "Toronto, ON",
         "work_type": "hybrid", "priorities": ["growth"]},
    ),
    SectionID.SKILL_GAP: (
        "Gaps: Kubernetes and Docker. Learning action: Kubernetes course plus a deployed project",
        {"job_sources": [{"label": "Shopify - Backend"}, {"label": "Wealthsimple - SWE"}],
         "requirements": [
             {"name": "Kubernetes", "priority": "must_have", "evidence": [
                 {"source_label": "Shopify - Backend", "quote": "Kubernetes in production"},
                 {"source_label": "Wealthsimple - SWE", "quote": "experience with K8s"}]},
             {"name": "Docker", "priority": "nice_to_have", "evidence": [
                 {"source_label": "Shopify - Backend", "quote": "Docker is a plus"}]},
             {"name": "Rust", "priority": "must_have", "evidence": [
                 {"source_label": "Invented Corp", "quote": "hallucinated"}]}]},
    ),
    SectionID.APPLICATION_STRATEGY: (
        "Rewrite resume, apply via referrals and company sites, 10 applications a week", {}),
    SectionID.INTERVIEW_PREP: (
        "Behavioral and system design interviews, 2 mock interviews a week", {}),
}

ROADMAP = """# Your Job Search Roadmap

## Where you are now
Backend developer with 3 years of Python and Django and an MSc in Computer Science.

## What you're aiming for
Senior backend roles in Toronto, hybrid, with growth as the priority.

## Skill gaps to close
- Kubernetes: 2 of 2 postings (Shopify - Backend, Wealthsimple - SWE)
- Docker: 1 of 2 postings (Shopify - Backend)

## Learning plan
Kubernetes course plus a deployed project.

## Application strategy
Rewrite resume, apply via referrals and company sites, 10 applications a week.

## Interview preparation
Behavioral and system design practice, 2 mock interviews a week.

## 8-week plan
| Week | Focus | Concrete targets |
| --- | --- | --- |
| 1 | Resume | Rewrite resume |
| 2 | Kubernetes | Start the course |
| 3 | Applications | 10 applications |
| 4 | Kubernetes | Deploy a project |
| 5 | Applications | 10 applications |
| 6 | Interviews | 2 mock interviews |
| 7 | Applications | 10 applications |
| 8 | Review | Adjust the plan |

## This week's checklist
- [ ] Rewrite resume summary
- [ ] Enrol in the Kubernetes course
- [ ] Shortlist 10 roles
- [ ] Ask two contacts for referrals
- [ ] Book one mock interview
"""


CALLS: list[tuple[str, str]] = []  # (kind, run name) for every model call in a conversation


class Recorder(FakeListChatModel):
    kind: str = ""
    fail_times: int = 0
    calls: list = Field(default_factory=list)

    async def ainvoke(self, input, config=None, **kwargs):
        self.calls.append(input)
        CALLS.append((self.kind, (config or {}).get("run_name", "")))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TimeoutError("provider timed out")
        return await super().ainvoke(input, config, **kwargs)


KINDS = {reply_mod: "reply", decision_mod: "decision", extraction_mod: "extraction",
         impl: "roadmap"}


def set_model(monkeypatch, module, text, fail_times=0):
    model = Recorder(responses=[text], kind=KINDS[module], fail_times=fail_times)
    monkeypatch.setattr(module, "get_chat_model", lambda config=None: model)
    return model


def count(kind: str) -> int:
    return sum(1 for k, _ in CALLS if k == kind)


@pytest.fixture
def complete_conversation(monkeypatch):
    """Run all five sections and return (final_state, roadmap_model, turns)."""

    async def run(fail_background_extraction_once: bool = False):
        CALLS.clear()
        monkeypatch.setattr(mu, "get_section_store", lambda: None)
        monkeypatch.setattr(impl, "get_roadmap_store", lambda: None)
        roadmap_model = set_model(monkeypatch, impl, ROADMAP)
        graph = build_xbuddy_graph()
        cfg = {"configurable": {"thread_id": "eval-full", "user_id": 1}}
        await graph.ainvoke({"messages": []}, cfg)

        state, turns = None, 0
        for section, (summary, extracted) in CONVERSATION.items():
            fields = get_section_template(section).required_fields
            # Turn 1: the user answers, the agent summarizes, a draft is saved.
            set_model(monkeypatch, reply_mod, f"Summary: {summary}. Does this look right?")
            set_model(monkeypatch, decision_mod, json.dumps({
                "router_directive": "stay", "should_save_content": True,
                "section_summary": summary, "covered_fields": fields}))
            state = await graph.ainvoke({"messages": [HumanMessage(summary)]}, cfg)
            assert state["current_section"] == section
            assert state["section_states"][section.value].confirmed_summary is None
            # Turn 2: the user confirms.
            set_model(monkeypatch, reply_mod, "Thanks, moving on.")
            set_model(monkeypatch, decision_mod, json.dumps({
                "router_directive": "next", "is_satisfied": True, "should_save_content": True,
                "section_summary": summary, "covered_fields": fields}))
            fail = 1 if fail_background_extraction_once and section == SectionID.BACKGROUND else 0
            set_model(monkeypatch, extraction_mod, json.dumps(extracted), fail_times=fail)
            state = await graph.ainvoke({"messages": [HumanMessage("Yes, that's right")]}, cfg)
            if fail:
                # The provider is back: the retry on the next turn re-extracts Background.
                set_model(monkeypatch, extraction_mod, json.dumps(extracted))
            turns += 2
        return state, roadmap_model, turns

    return run


@pytest.mark.asyncio
async def test_complete_conversation_produces_a_grounded_roadmap(complete_conversation):
    state, roadmap_model, turns = await complete_conversation()
    roadmap, data = state["roadmap"], state["user_data"]

    # --- Cost: the model calls of the whole conversation, with no hidden retries -------------
    assert turns == 10
    assert (count("reply"), count("decision"), count("roadmap")) == (10, 10, 1)
    extraction_runs = [name for kind, name in CALLS if kind == "extraction"]
    assert extraction_runs == ["extract[background]", "extract[target_role]", "extract[skill_gap]"]
    assert len(CALLS) == 24

    # --- Completion: every section confirmed in two turns each -------------------------------
    assert state["finished"] is True
    assert all(s.status == SectionStatus.DONE for s in state["section_states"].values())
    for section, (summary, _) in CONVERSATION.items():
        assert state["section_states"][section.value].confirmed_summary == summary
    assert state.get("error_count", 0) == 0

    # --- Structured data: extracted, traceable, hallucination dropped --------------------------
    assert data.profile.current_role == "Backend Developer"
    assert data.target_role.location_region == "Toronto, ON"
    assert [s.label for s in data.job_sources] == ["Shopify - Backend", "Wealthsimple - SWE"]
    names = {r.name: r.frequency for r in data.requirements}
    assert names == {"Kubernetes": 2, "Docker": 1}  # "Rust" cited a posting the user never gave

    # --- Quality criteria: structure, grounding, completeness, consistency, safety ------------
    assert validate_roadmap(roadmap, data) == []
    assert find_contradictions(roadmap, data) == []  # 3 years, hybrid, must-have vs nice-to-have
    for heading in REQUIRED_HEADINGS:
        assert heading in roadmap
    assert "Kubernetes: 2 of 2 postings" in roadmap
    assert "Rust" not in roadmap
    assert roadmap == state["messages"][-1].content  # delivered to the user as the last message

    # --- Grounding of the model's input: confirmed summaries and traceable requirements only ---
    assert len(roadmap_model.calls) == 1  # passed validation first time, no rewrite needed
    given = roadmap_model.calls[0][1].content
    for summary, _ in CONVERSATION.values():
        assert summary in given
    assert "Kubernetes (must-have; 2 of 2 postings: Shopify - Backend, Wealthsimple - SWE)" in given
    assert "Invented Corp" not in given
    assert "Rust" not in given


@pytest.mark.asyncio
async def test_transient_extraction_failure_costs_exactly_one_visible_retry(complete_conversation):
    """The provider fails once while Background is confirmed. The conversation still
    completes, the profile is recovered, and the extra cost is one labelled call."""
    state, _, turns = await complete_conversation(fail_background_extraction_once=True)

    extraction_runs = [name for kind, name in CALLS if kind == "extraction"]
    assert extraction_runs == [
        "extract[background]",            # failed
        "extract[background] retry 2/3",  # recovered on the next turn
        "extract[target_role]",
        "extract[skill_gap]",
    ]
    assert len(CALLS) == 25  # one more call than the clean run, nothing hidden
    assert turns == 10
    assert state["finished"] is True
    assert state["user_data"].profile.current_role == "Backend Developer"
    assert state["section_states"]["background"].unverified_fields == []
    assert state["section_states"]["background"].extraction_attempts == 2
    assert validate_roadmap(state["roadmap"], state["user_data"]) == []

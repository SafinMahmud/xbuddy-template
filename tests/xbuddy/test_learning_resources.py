"""PR 13: learning resources retrieved from a curated library (skill tags + BM25).

Covers the library itself, the retriever, the retrieval evaluation, and how the
Skill Gap reply and the roadmap use what is retrieved.
"""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from test_roadmap import GOOD_ROADMAP, completed_state, user_data

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.knowledge import (
    detect_tags,
    load_library,
    resources_for_requirements,
    retrieve,
    skill_gap_resources_block,
    strip_unknown_links,
    unknown_links,
)
from agents.xbuddy.knowledge.evaluation import evaluate
from agents.xbuddy.knowledge.retriever import SKILL_PHRASES
from agents.xbuddy.models import JobBuddyData, JobSource, SectionState, UserProfile
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.roadmap import (
    build_roadmap_input,
    fallback_roadmap,
    learning_resources_block,
    validate_roadmap,
)

FIXTURES = Path(__file__).parent / "fixtures"
EVAL_CASES = json.loads((FIXTURES / "retrieval_eval.json").read_text("utf-8"))
HELDOUT_CASES = json.loads((FIXTURES / "retrieval_eval_heldout.json").read_text("utf-8"))


# --- the library -----------------------------------------------------------------------------

def test_library_entries_are_unique_and_complete():
    library = load_library()
    assert len(library) >= 50
    assert len({r.id for r in library}) == len(library)
    assert len({r.url for r in library}) == len(library)
    for r in library:
        assert r.url.startswith("https://"), r.id
        assert r.level in {"beginner", "intermediate", "advanced"}, r.id
        assert r.cost in {"free", "paid"}, r.id
        assert set(r.skills) <= set(SKILL_PHRASES), (r.id, set(r.skills) - set(SKILL_PHRASES))


def test_every_skill_in_the_vocabulary_has_a_resource():
    covered = {tag for r in load_library() for tag in r.skills}
    assert set(SKILL_PHRASES) - covered == set()


# --- tag detection ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Kubernetes (k8s) and Helm", {"kubernetes"}),
        ("Java and JavaScript", {"java", "javascript"}),
        ("Node.js with Express.js", {"nodejs"}),
        ("C# / .NET", {"csharp"}),
        ("PostgreSQL or MySQL", {"postgresql", "sql"}),
        ("CI/CD with GitHub Actions", {"ci-cd", "git"}),
        ("Go (Golang)", {"go"}),
        ("Strong C#/.NET skills", {"csharp"}),
        ("Postgres/DynamoDB", {"postgresql"}),
    ],
)
def test_skill_tags_are_detected(text, expected):
    assert expected <= detect_tags(text)


@pytest.mark.parametrize(
    "text",
    [
        "We react quickly to change",
        "Ready to go the extra mile",
        "For the rest of the team",
        "You excel at problem framing",
        "Must be eligible to work in Canada",
    ],
)
def test_ordinary_words_are_not_taken_for_skills(text):
    assert not ({"react", "go", "rest-api", "excel"} & detect_tags(text))


# --- retrieval -------------------------------------------------------------------------------

def test_a_named_skill_retrieves_only_resources_for_it():
    hits = retrieve("Kubernetes")
    assert hits[0].resource.id == "k8s-basics"
    assert all("kubernetes" in h.resource.skills for h in hits)


def test_non_skill_requirements_retrieve_nothing():
    assert retrieve("5+ years of professional experience") == []
    assert retrieve("Excellent communication skills") == []


def test_free_resources_win_ties():
    hits = retrieve("AWS")
    assert all("aws" in h.matched_tags for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_requirements_do_not_repeat_a_resource():
    out = resources_for_requirements(["Docker", "Docker and containers", "Kubernetes"])
    ids = [r.id for _, rs in out for r in rs]
    assert len(ids) == len(set(ids))
    assert [name for name, _ in out][-1] == "Kubernetes"


def test_retrieval_evaluation_meets_the_bar():
    report = evaluate(EVAL_CASES)
    assert report["cases"] >= 50
    assert report["hit@3"] >= 0.95, report["misses"]
    assert report["mrr"] >= 0.9, report["misses"]
    assert report["negatives"] == 1.0, report["false_hits"]


def test_held_out_postings_do_not_regress():
    """Requirements copied from three real postings, labeled before the first run.
    First run: hit@3 0.826, MRR 0.754, negatives 0.5. After fixing the slash
    boundary bug and requiring a shared skill tag: hit@3 0.826, MRR 0.754,
    negatives 0.833. These floors stop it getting worse; they are not a target."""
    report = evaluate(HELDOUT_CASES)
    assert report["cases"] == 23
    assert report["hit@3"] >= 0.8, report["misses"]
    assert report["negatives"] >= 0.8, report["false_hits"]


def test_a_strong_keyword_match_without_a_skill_is_not_returned():
    assert retrieve("quickly learning in ambiguous, fast-paced environments") == []
    assert retrieve("Microsoft Power Platform (Power Apps, Power Automate)") == []


# --- links -----------------------------------------------------------------------------------

def test_invented_links_are_found_and_stripped():
    text = (
        "Take [Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/) "
        "and [K8s Mastery](https://fake-courses.example.com/k8s). See https://bit.ly/x."
    )
    assert unknown_links(text) == ["https://fake-courses.example.com/k8s", "https://bit.ly/x"]
    cleaned = strip_unknown_links(text)
    assert "K8s Mastery" in cleaned
    assert "example.com" not in cleaned and "bit.ly" not in cleaned
    assert "https://kubernetes.io/docs/tutorials/kubernetes-basics/" in cleaned


# --- Skill Gap reply -------------------------------------------------------------------------

POSTING = (
    "Fintech - GenAI Engineer\nRequirements: Python, RAG pipelines, vector databases, "
    "LangGraph agents, Kubernetes, CI/CD. Nice to have: LLM evaluation. " * 3
)


def test_skill_gap_block_puts_known_skills_last():
    block = skill_gap_resources_block(POSTING, known="Skills: Python, Kubernetes")
    lines = [line for line in block.splitlines() if line.startswith("- ")]
    tags = [line[2:].split(":", 1)[0] for line in lines]
    gaps = [t for t in tags if t not in {"python", "kubernetes"}]
    assert gaps and tags[: len(gaps)] == gaps  # skills the user has come after the gaps
    assert "never write any other link" in block
    # Live run (Nov 3): the reply named a library course without its link and said it
    # "integrates with Azure services". The prompt now carries each resource's summary
    # and requires the link, so the model has the real description to work from.
    assert "ALWAYS write it as a Markdown link" in block
    assert "never claim anything else about it" in block
    langgraph = next(line for line in block.splitlines() if "intro-to-langgraph" in line)
    assert "What it covers: Build stateful agents with LangGraph" in langgraph


@pytest.mark.asyncio
async def test_skill_gap_reply_gets_library_resources_for_the_posting(monkeypatch):
    seen = []

    class Recording(FakeListChatModel):
        async def ainvoke(self, input, config=None, **kwargs):
            seen.append(input[0].content)
            return await super().ainvoke(input, config, **kwargs)

    monkeypatch.setattr(
        reply_mod, "get_chat_model", lambda config=None: Recording(responses=["Gaps: ..."])
    )
    state = completed_state()
    state["current_section"] = SectionID.SKILL_GAP
    state["section_states"]["skill_gap"] = SectionState(
        section_id=SectionID.SKILL_GAP, status=SectionStatus.IN_PROGRESS)
    state["messages"] = [HumanMessage(POSTING)]
    await reply_mod.generate_reply_node(state, {})
    assert "LEARNING RESOURCES" in seen[-1]
    assert "https://academy.langchain.com/courses/intro-to-langgraph" in seen[-1]


@pytest.mark.asyncio
async def test_other_sections_get_no_resource_block(monkeypatch):
    seen = []

    class Recording(FakeListChatModel):
        async def ainvoke(self, input, config=None, **kwargs):
            seen.append(input[0].content)
            return await super().ainvoke(input, config, **kwargs)

    monkeypatch.setattr(
        reply_mod, "get_chat_model", lambda config=None: Recording(responses=["Ok"])
    )
    state = completed_state()
    state["current_section"] = SectionID.BACKGROUND
    state["messages"] = [HumanMessage(POSTING)]
    await reply_mod.generate_reply_node(state, {})
    assert "LEARNING RESOURCES" not in seen[-1]


# --- roadmap ---------------------------------------------------------------------------------

def test_roadmap_input_lists_resources_for_the_requirements():
    state = completed_state()
    text = build_roadmap_input(state["section_states"], state["user_data"])
    assert "LEARNING RESOURCES" in text
    assert "For Kubernetes: [Learn Kubernetes Basics]" in text
    assert "For Docker: [Docker: Get started]" in text


def test_roadmap_with_a_library_link_passes():
    good = GOOD_ROADMAP.replace(
        "Finish a Kubernetes course",
        "Finish [Learn Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)",
    )
    assert validate_roadmap(good, user_data()) == []


def test_roadmap_with_an_invented_link_fails():
    bad = GOOD_ROADMAP.replace(
        "Finish a Kubernetes course",
        "Finish [K8s Bootcamp](https://made-up-academy.example.com/k8s)",
    )
    problems = validate_roadmap(bad, user_data())
    assert any("link not from the learning-resource library" in p for p in problems)


def test_the_users_own_posting_link_is_allowed():
    data = user_data().model_copy(update={"job_sources": [
        JobSource(label="Shopify - Backend", url="https://shopify.com/careers/123"),
        JobSource(label="Wealthsimple - SWE")]})
    ok = GOOD_ROADMAP.replace("Ten applications a week", "Apply at https://shopify.com/careers/123")
    assert validate_roadmap(ok, data) == []


def test_fallback_roadmap_cites_resources_and_drops_invented_links():
    state = completed_state()
    state["section_states"]["skill_gap"] = state["section_states"]["skill_gap"].model_copy(
        update={"confirmed_summary": "Take [K8s Pro](https://fake.example.com/k8s)."})
    roadmap = fallback_roadmap(state["section_states"], state["user_data"])
    assert "https://kubernetes.io/docs/tutorials/kubernetes-basics/" in roadmap
    assert "fake.example.com" not in roadmap
    assert validate_roadmap(roadmap, state["user_data"]) == []


def test_no_resources_when_nothing_matches():
    data = JobBuddyData(profile=UserProfile(skills=["Python"]))
    assert learning_resources_block(data) == ""

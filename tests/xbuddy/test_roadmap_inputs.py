"""PR 5 tests: the roadmap when the collected inputs are missing or contradictory."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field

from agents.xbuddy.enums import RequirementPriority, SectionID, SectionStatus, WorkType
from agents.xbuddy.models import (
    JobBuddyData,
    JobRequirement,
    JobSource,
    RequirementEvidence,
    SectionState,
    TargetRole,
    UserProfile,
    default_state,
)
from agents.xbuddy.nodes import implementation as impl
from agents.xbuddy.nodes.implementation import implementation_node
from agents.xbuddy.roadmap import (
    build_roadmap_input,
    conflicting_inputs,
    fallback_roadmap,
    missing_inputs,
    reconcile,
    stated_total_years,
    validate_roadmap,
)

CONFIG = {"configurable": {"thread_id": "t", "user_id": 1}}
SOURCES = ["Shopify - Backend", "Wealthsimple - SWE"]
SUMMARIES = {
    SectionID.BACKGROUND: "Backend developer with 3 years of experience in Python.",
    SectionID.TARGET_ROLE: "Senior backend roles in Toronto, hybrid.",
    SectionID.SKILL_GAP: "Gap: Kubernetes. Learning action: a Kubernetes course.",
    SectionID.APPLICATION_STRATEGY: "10 applications a week through referrals.",
    SectionID.INTERVIEW_PREP: "Behavioral and system design, 2 mock interviews a week.",
}


def roadmap(where="Backend developer with 3 years of experience in Python.",
            aim="Senior backend roles in Toronto, hybrid.",
            gaps="- Kubernetes: 2 of 2 postings (Shopify - Backend, Wealthsimple - SWE)") -> str:
    weeks = "\n".join(f"| {n} | Focus {n} | Target {n} |" for n in range(1, 9))
    checklist = "\n".join(f"- [ ] Task {n}" for n in range(1, 6))
    return f"""# Your Job Search Roadmap

## Where you are now
{where}

## What you're aiming for
{aim}

## Skill gaps to close
{gaps}

## Learning plan
A Kubernetes course and one deployed project.

## Application strategy
10 applications a week through referrals.

## Interview preparation
2 mock interviews a week.

## 8-week plan
| Week | Focus | Concrete targets |
| --- | --- | --- |
{weeks}

## This week's checklist
{checklist}
"""


def sections(skip=(), unverified=None) -> dict:
    states = default_state()["section_states"]
    for sid, text in SUMMARIES.items():
        if sid in skip:
            continue
        states[sid.value] = SectionState(
            section_id=sid, status=SectionStatus.DONE, confirmed_summary=text,
            unverified_fields=(unverified or {}).get(sid, []))
    return states


def data(years=3, work_type=WorkType.HYBRID, with_postings=True) -> JobBuddyData:
    requirement = JobRequirement(
        name="Kubernetes", priority=RequirementPriority.MUST_HAVE,
        evidence=[RequirementEvidence(source_label=s, quote="Kubernetes") for s in SOURCES])
    return JobBuddyData(
        profile=UserProfile(current_role="Backend Developer", years_experience=years),
        target_role=TargetRole(titles=["Senior Backend Developer"], work_type=work_type),
        job_sources=[JobSource(label=s) for s in SOURCES] if with_postings else [],
        requirements=[requirement] if with_postings else [],
    )


class Scripted(FakeListChatModel):
    calls: list = Field(default_factory=list)

    async def ainvoke(self, input, config=None, **kwargs):
        self.calls.append(input)
        return await super().ainvoke(input, config, **kwargs)


def use(monkeypatch, *responses) -> Scripted:
    model = Scripted(responses=list(responses))
    monkeypatch.setattr(impl, "get_chat_model", lambda config=None: model)
    monkeypatch.setattr(impl, "get_roadmap_store", lambda: None)
    return model


def state(section_states, user_data) -> dict:
    return {"messages": [], "thread_id": "t", "user_id": 1, **default_state(),
            "section_states": section_states, "user_data": user_data,
            "current_section": SectionID.INTERVIEW_PREP}


# --- missing inputs --------------------------------------------------------------------------

def test_complete_inputs_report_nothing_missing_or_conflicting():
    assert missing_inputs(sections(), data()) == []
    assert conflicting_inputs(sections(), data()) == []
    text = build_roadmap_input(sections(), data())
    assert "MISSING INFORMATION" not in text
    assert "CONFLICTING INFORMATION" not in text


def test_each_kind_of_missing_input_is_reported():
    assert missing_inputs(sections(skip=[SectionID.INTERVIEW_PREP]), data()) == [
        "Interview Preparation was not confirmed"]
    assert any("no job postings were supplied" in m
               for m in missing_inputs(sections(), data(with_postings=False)))
    no_requirements = data().model_copy(update={"requirements": []})
    assert any("no requirements could be extracted" in m
               for m in missing_inputs(sections(), no_requirements))
    unverified = sections(unverified={SectionID.BACKGROUND: ["education"]})
    assert "Background: could not be verified as data: education" in missing_inputs(
        unverified, data())


def test_model_is_told_what_is_missing_and_not_to_invent_it():
    text = build_roadmap_input(sections(skip=[SectionID.APPLICATION_STRATEGY]),
                               data(with_postings=False))
    assert "MISSING INFORMATION" in text
    assert "do not invent it" in text
    assert "- Application Strategy was not confirmed" in text
    assert "no job postings were supplied" in text
    assert "10 applications a week" not in text  # an unconfirmed section is never sent


def test_posting_counts_are_rejected_when_no_postings_were_supplied():
    invented = roadmap(gaps="- Kubernetes: 4 of 5 postings mention it")
    problems = validate_roadmap(invented, data(with_postings=False))
    assert any("ungrounded claim '4 of 5 postings'" in p for p in problems)

    honest = roadmap(gaps="No job postings were shared, so skill gaps were not analysed.")
    assert validate_roadmap(honest, data(with_postings=False)) == []


def test_fallback_states_what_was_not_covered():
    states = sections(skip=[SectionID.INTERVIEW_PREP])
    text = fallback_roadmap(states, data(with_postings=False))
    assert "Not covered in this conversation." in text
    assert validate_roadmap(text, data(with_postings=False)) == []


@pytest.mark.asyncio
async def test_roadmap_with_missing_inputs_is_generated_without_invented_counts(monkeypatch):
    honest = roadmap(gaps="No job postings were shared, so skill gaps were not analysed.")
    model = use(monkeypatch, honest)
    result = await implementation_node(
        state(sections(), data(with_postings=False)), CONFIG)

    assert result["roadmap"] == honest.strip()
    assert "last_error" not in result
    assert "MISSING INFORMATION" in model.calls[0][1].content


# --- contradictory inputs --------------------------------------------------------------------

def test_conflicts_between_confirmed_summary_and_extracted_data_are_found():
    conflicts = conflicting_inputs(sections(), data(years=5, work_type=WorkType.REMOTE))
    assert conflicts == [
        "years of experience: the confirmed summary says 3, the extracted profile says 5",
        "work arrangement: the confirmed summary says hybrid, the extracted target says remote",
    ]


def test_confirmed_summary_wins_when_inputs_conflict():
    conflicted = data(years=5, work_type=WorkType.REMOTE)
    resolved = reconcile(sections(), conflicted)

    assert resolved.profile.years_experience == 3
    assert resolved.target_role.work_type == WorkType.HYBRID
    assert conflicted.profile.years_experience == 5  # the stored data is not modified
    assert reconcile(sections(), data()) == data()  # no conflict, nothing changes


def test_model_is_told_about_the_conflict_and_which_value_to_use():
    text = build_roadmap_input(sections(), data(years=5))
    assert "CONFLICTING INFORMATION" in text
    assert "use the confirmed summary" in text
    assert "the confirmed summary says 3, the extracted profile says 5" in text


@pytest.mark.asyncio
async def test_roadmap_following_the_confirmed_summary_is_accepted(monkeypatch):
    """Without reconciling, '3 years' would be rejected against the extracted '5'."""
    model = use(monkeypatch, roadmap())
    result = await implementation_node(
        state(sections(), data(years=5, work_type=WorkType.REMOTE)), CONFIG)

    assert result["roadmap"] == roadmap().strip()
    assert "last_error" not in result
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_roadmap_repeating_the_conflicting_value_is_rewritten(monkeypatch):
    wrong = roadmap(where="Backend developer with 5 years of experience in Python.",
                    aim="Senior backend roles in Toronto, fully remote.")
    model = use(monkeypatch, wrong, roadmap())
    result = await implementation_node(
        state(sections(), data(years=5, work_type=WorkType.REMOTE)), CONFIG)

    assert result["roadmap"] == roadmap().strip()
    feedback = model.calls[1][-1].content
    assert "says '5 years' of experience, the user confirmed 3" in feedback
    assert "targets remote work, the user confirmed hybrid" in feedback


# --- years of experience: total vs tenure in one role ------------------------------------------

def background(text: str) -> dict:
    states = sections()
    states["background"] = SectionState(
        section_id=SectionID.BACKGROUND, status=SectionStatus.DONE, confirmed_summary=text)
    return states


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Backend developer with 3 years of experience.", 3),
        ("5+ years of professional experience in Python.", 5),
        ("Experience: 4 years. Skills: Python.", 4),
        ("- **Experience:** 6 years", 6),
        ("Backend developer, 3 years, Python.", 3),  # one lone number is the total
        ("5 years of experience, including 2 years as team lead.", 5),  # explicit total wins
        ("3 years at Acme and 2 years at Beta.", None),  # two tenures, no stated total
        ("8 years of experience overall, 5 years of experience in backend.", None),  # two totals
        ("Backend developer, Python and Django.", None),
    ],
)
def test_total_experience_is_read_only_when_it_is_clear(text, expected):
    assert stated_total_years(text) == expected


def test_tenure_in_a_prior_role_is_not_mistaken_for_total_experience():
    """The summary states the total and a tenure. The extracted total matches the total."""
    states = background("5 years of experience, including 2 years as team lead.")
    assert conflicting_inputs(states, data(years=5)) == []
    assert reconcile(states, data(years=5)).profile.years_experience == 5


def test_wrong_extraction_is_corrected_from_the_explicit_total_not_the_tenure():
    """Extraction picked up the tenure (2). The summary's explicit total (5) wins."""
    states = background("5 years of experience, including 2 years as team lead.")
    assert conflicting_inputs(states, data(years=2)) == [
        "years of experience: the confirmed summary says 5, the extracted profile says 2"]
    assert reconcile(states, data(years=2)).profile.years_experience == 5


def test_ambiguous_summary_is_not_reconciled_and_keeps_the_extracted_total():
    """Several durations and no stated total: no guess is made either way."""
    states = background("3 years at Acme and 2 years at Beta.")
    assert conflicting_inputs(states, data(years=5)) == []
    assert reconcile(states, data(years=5)).profile.years_experience == 5
    assert reconcile(states, data(years=5)) == data(years=5)


def test_roadmap_may_mention_a_tenure_alongside_the_total():
    text = roadmap(where="Backend developer with 5 years of experience, including 2 years "
                         "as team lead.")
    assert validate_roadmap(text, data(years=5)) == []

    wrong_total = roadmap(where="Backend developer with 7 years of experience, including "
                                "2 years as team lead.")
    problems = validate_roadmap(wrong_total, data(years=5))
    assert problems == ["contradiction: roadmap says '7 years' of experience, the user confirmed 5"]

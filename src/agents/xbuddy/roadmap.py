"""Job search roadmap: inputs, prompt, quality checks, and a deterministic fallback.

Quality criteria (validate_roadmap) every delivered roadmap must meet:
  Structure  all required headings, an 8-week plan table with 8 week rows, and a
             checklist of at least 5 items.
  Grounding  every "N of M postings" claim matches the user's real posting count,
             and the most-demanded extracted requirements are named.
  Completeness  no required section is left empty.
  Consistency   no contradiction with confirmed data: years of experience, the
             work arrangement, and whether a requirement is must-have or
             nice-to-have.
  Safety     no placeholder text and no salary figures.
The fallback roadmap has the required structure and grounding by construction.
"""

import re

from .enums import SectionID, SectionStatus
from .models import JobBuddyData, SectionState
from .prompts import get_section_template

ROADMAP_PROMPT = """You are JobBuddy. Write the user's personalized job search roadmap in
Markdown, using ONLY the confirmed information below. Do not invent facts.

Use exactly these sections:
# Your Job Search Roadmap
## Where you are now
## What you're aiming for
## Skill gaps to close
(ranked by how many of the user's job postings mention each requirement; name the
postings for each gap)
## Learning plan
## Application strategy
## Interview preparation
## 8-week plan
(a table: Week | Focus | Concrete targets, built from the weekly targets above)
## This week's checklist
(5 to 8 checkbox items the user can start today)

Rules:
- The 8-week plan table must have exactly 8 week rows (Week 1 to Week 8).
- When you say how many postings mention a skill, use the counts given below
  exactly, written as "N of M postings".
- Name every extracted requirement listed below under "Skill gaps to close".
- No placeholders such as [TBD]. No salary or pay figures.

Keep it concise, specific, and encouraging.
"""

REQUIRED_HEADINGS = [
    "# Your Job Search Roadmap",
    "## Where you are now",
    "## What you're aiming for",
    "## Skill gaps to close",
    "## Learning plan",
    "## Application strategy",
    "## Interview preparation",
    "## 8-week plan",
    "## This week's checklist",
]
PLAN_WEEKS = 8
MIN_CHECKLIST_ITEMS = 5
TOP_REQUIREMENTS = 5
_PLACEHOLDERS = ("[tbd]", "[not provided]", "todo", "lorem ipsum")
_COUNT_CLAIM = re.compile(r"(\d+)\s+of\s+(\d+)\s+postings?", re.IGNORECASE)
_MONEY = re.compile(r"[$£€]\s?\d")
_TABLE_DIVIDER = re.compile(r"^\|?[\s:|-]+\|?$")
_YEARS = re.compile(r"(\d+)\+?\s*(?:years|year|yrs)\b", re.IGNORECASE)
MIN_SECTION_CHARS = 10
_WORK_WORDS = {
    "remote": ("remote",),
    "hybrid": ("hybrid",),
    "onsite": ("onsite", "on-site", "in-office", "in office"),
}
_PRIORITY_WORDS = {
    "must_have": ("nice-to-have", "nice to have", "optional"),
    "nice_to_have": ("must-have", "must have"),
}


def gather_confirmed(section_states: dict[str, SectionState]) -> dict[SectionID, str]:
    """Confirmed summaries of done sections, in section order."""
    confirmed = {}
    for sid in SectionID:
        state = section_states.get(sid.value)
        if state and state.status == SectionStatus.DONE and state.confirmed_summary:
            confirmed[sid] = state.confirmed_summary
    return confirmed


def requirements_table(user_data: JobBuddyData) -> str:
    """Traceable requirement list, strongest signal first."""
    total = len(user_data.job_sources)
    rows = sorted(user_data.requirements, key=lambda r: (-r.frequency, r.name.lower()))
    lines = []
    for r in rows:
        sources = ", ".join(sorted({e.source_label for e in r.evidence}))
        label = r.priority.value.replace("_", "-")
        lines.append(f"- {r.name} ({label}; {r.frequency} of {total} postings: {sources})")
    return "\n".join(lines)


def build_roadmap_input(section_states: dict[str, SectionState], user_data: JobBuddyData) -> str:
    parts = [
        f"{get_section_template(sid).name}:\n{text}"
        for sid, text in gather_confirmed(section_states).items()
    ]
    table = requirements_table(user_data)
    if table:
        parts.append("Extracted requirements (with source postings):\n" + table)
    return "\n\n".join(parts)


FALLBACK_CHECKLIST = """- [ ] Update your resume using the application strategy above
- [ ] Update your LinkedIn headline and summary
- [ ] Start the first learning action for your top skill gap
- [ ] Send this week's applications and outreach
- [ ] Practise one interview story out loud"""


def fallback_roadmap(section_states: dict[str, SectionState], user_data: JobBuddyData) -> str:
    """Plain roadmap built without the model. It meets validate_roadmap by construction,
    so the user always receives a complete, grounded output."""
    confirmed = gather_confirmed(section_states)

    def said(sid: SectionID) -> str:
        return confirmed.get(sid, "Not covered in this conversation.")

    gaps = requirements_table(user_data) or said(SectionID.SKILL_GAP)
    focus = [
        "Resume and LinkedIn refresh",
        "Start the first learning action",
        "Applications and outreach",
        "Learning and a portfolio example",
        "Applications and networking follow-ups",
        "Interview practice",
        "Applications and mock interviews",
        "Review progress and adjust the plan",
    ]
    plan = ["| Week | Focus | Concrete targets |", "| --- | --- | --- |"]
    plan += [
        f"| {week} | {item} | Follow your application strategy and learning plan |"
        for week, item in enumerate(focus, start=1)
    ]
    parts = [
        "# Your Job Search Roadmap",
        "## Where you are now", said(SectionID.BACKGROUND),
        "## What you're aiming for", said(SectionID.TARGET_ROLE),
        "## Skill gaps to close", gaps,
        "## Learning plan", said(SectionID.SKILL_GAP),
        "## Application strategy", said(SectionID.APPLICATION_STRATEGY),
        "## Interview preparation", said(SectionID.INTERVIEW_PREP),
        "## 8-week plan", "\n".join(plan),
        "## This week's checklist",
        FALLBACK_CHECKLIST,
    ]
    return "\n\n".join(parts)

def _normalize(text: str) -> str:
    return text.replace("\u2019", "'").lower()


def _section_body(roadmap: str, heading: str) -> str:
    """Text under a heading, up to the next heading of the same or higher level."""
    lines = roadmap.replace("\u2019", "'").splitlines()
    body, inside = [], False
    for line in lines:
        if line.strip().lower() == heading.lower():
            inside = True
            continue
        if inside and line.startswith("#"):
            break
        if inside:
            body.append(line)
    return "\n".join(body)


def ranked_requirements(user_data: JobBuddyData) -> list:
    return sorted(user_data.requirements, key=lambda r: (-r.frequency, r.name.lower()))


def validate_roadmap(roadmap: str, user_data: JobBuddyData) -> list[str]:
    """Return the quality problems found in a roadmap. An empty list means it passes."""
    problems: list[str] = []
    text = _normalize(roadmap or "")
    if not text.strip():
        return ["roadmap is empty"]

    # Structure
    lines = {line.strip() for line in text.splitlines()}
    for heading in REQUIRED_HEADINGS:
        if heading.lower() not in lines:
            problems.append(f"missing section: {heading}")
    plan_rows = [
        line for line in _section_body(roadmap, "## 8-week plan").splitlines()
        if line.strip().startswith("|") and not _TABLE_DIVIDER.match(line.strip())
    ]
    week_rows = max(len(plan_rows) - 1, 0)  # minus the header row
    if week_rows != PLAN_WEEKS:
        problems.append(f"8-week plan has {week_rows} week rows, expected {PLAN_WEEKS}")
    checklist = _section_body(roadmap, "## This week's checklist").count("- [ ]")
    if checklist < MIN_CHECKLIST_ITEMS:
        problems.append(f"checklist has {checklist} items, expected at least {MIN_CHECKLIST_ITEMS}")

    # Grounding
    total = len(user_data.job_sources)
    for match in _COUNT_CLAIM.finditer(roadmap):
        n, m = int(match.group(1)), int(match.group(2))
        if m != total or n > m:
            problems.append(
                f"ungrounded claim '{match.group(0)}': the user supplied {total} postings"
            )
    for requirement in ranked_requirements(user_data)[:TOP_REQUIREMENTS]:
        if requirement.name.lower() not in text:
            problems.append(f"skill gap not mentioned: {requirement.name}")

    # Completeness
    for heading in REQUIRED_HEADINGS[1:]:
        if heading.lower() in lines and len(_section_body(roadmap, heading).strip()) < MIN_SECTION_CHARS:
            problems.append(f"empty section: {heading}")

    # Consistency with confirmed data
    problems.extend(find_contradictions(roadmap, user_data))

    # Safety
    for placeholder in _PLACEHOLDERS:
        if placeholder in text:
            problems.append(f"placeholder text: {placeholder}")
    if _MONEY.search(roadmap):
        problems.append("contains salary or money figures")
    return problems


def find_contradictions(roadmap: str, user_data: JobBuddyData) -> list[str]:
    """Statements in the roadmap that conflict with what the user confirmed."""
    problems: list[str] = []

    years = user_data.profile.years_experience
    if years is not None:
        for match in _YEARS.finditer(_section_body(roadmap, "## Where you are now")):
            if int(match.group(1)) != years:
                problems.append(
                    f"contradiction: roadmap says '{match.group(0)}' of experience, "
                    f"the user confirmed {years}"
                )

    work_type = user_data.target_role.work_type
    if work_type is not None and work_type.value in _WORK_WORDS:
        aim = _section_body(roadmap, "## What you're aiming for").lower()
        mentioned = {k for k, words in _WORK_WORDS.items() if any(w in aim for w in words)}
        if mentioned and work_type.value not in mentioned:
            problems.append(
                f"contradiction: roadmap targets {', '.join(sorted(mentioned))} work, "
                f"the user confirmed {work_type.value}"
            )

    gap_lines = _section_body(roadmap, "## Skill gaps to close").lower().splitlines()
    for requirement in user_data.requirements:
        wrong_words = _PRIORITY_WORDS[requirement.priority.value]
        for line in gap_lines:
            if requirement.name.lower() in line and any(w in line for w in wrong_words):
                problems.append(
                    f"contradiction: '{requirement.name}' is "
                    f"{requirement.priority.value.replace('_', '-')} in the postings, "
                    "but the roadmap describes it differently"
                )
                break
    return problems

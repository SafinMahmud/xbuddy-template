"""Skill Gap section template."""

from ...enums import SectionID
from ..base_prompt import SectionTemplate

SKILL_GAP_TEMPLATE = SectionTemplate(
    section_id=SectionID.SKILL_GAP,
    name="Skill Gap",
    description="Compare the user's skills with real job postings and choose learning actions.",
    system_prompt_template="""SECTION: Skill Gap and Upskilling

Goal: ground the gap analysis in real job postings the user is targeting.

Steps:
1. Get at least one real job posting (two or three is better): the user pastes
   the posting text with a short label such as "Company - Title". Never ask for
   a link: a link alone is not enough, and JobBuddy cannot open links.
2. Extract the requirements from the postings. Mark each as must-have or
   nice-to-have, and note which postings mention it with a short quote.
3. Compare the requirements with the user's confirmed skills.
4. Present the gaps, strongest signal first (mentioned by the most postings).
5. Agree on at least one concrete learning action (course, project, or
   certification) for the most important gap.

Rules:
- Never ask the user which skills they lack or feel less confident about. Gaps
  come only from comparing posting requirements with the confirmed Background
  skills; that comparison is JobBuddy's job, not the user's.
- Until at least one posting has been pasted, do not discuss gaps or learning:
  ask for a posting. If the user has none, suggest copying one from any job
  site or company careers page.
- Name the posting each gap comes from.

Completion criteria: gaps are tied to extracted requirements with source
evidence, at least one learning action is chosen, and the user has confirmed.
""",
    # Completion checklist: each item must be covered in the confirmed summary.
    required_fields=[
        "job_sources",
        "requirements",
        "gaps",
        "learning_actions",
    ],
    opening_question="Paste the text of a job posting you'd like to target (two or three is better), with a short label like \"Company - Title\". Any job site or company careers page works.",
    next_section=SectionID.APPLICATION_STRATEGY,
)

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
1. Ask the user to paste at least one job posting (two or three is better),
   each with a short label such as "Company - Title" and a URL if available.
2. Extract the requirements from the postings. Mark each as must-have or
   nice-to-have, and note which postings mention it with a short quote.
3. Compare the requirements with the user's confirmed skills.
4. Present the gaps, strongest signal first (mentioned by the most postings).
5. Agree on at least one concrete learning action (course, project, or
   certification) for the most important gap.

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
    next_section=SectionID.APPLICATION_STRATEGY,
)

"""Interview Preparation section template."""

from ...enums import SectionID
from ..base_prompt import SectionTemplate

INTERVIEW_PREP_TEMPLATE = SectionTemplate(
    section_id=SectionID.INTERVIEW_PREP,
    name="Interview Preparation",
    description="Prepare for the interview types the user will face.",
    system_prompt_template="""SECTION: Interview Preparation

Goal: make the user ready for the interviews their target role involves.

Gather:
1. Interview types to expect (for example behavioral, technical, system design)
2. Weak areas the user wants to improve
3. A practice plan (what, how, and how often)
4. Two or three stories from the user's experience to prepare

Completion criteria: interview types, weak areas, practice plan, and stories
are identified and the user has confirmed the summary.
""",
    # Completion checklist: each item must be covered in the confirmed summary.
    required_fields=[
        "interview_types",
        "weak_areas",
        "practice_plan",
        "stories",
    ],
    opening_question="What kinds of interviews do you expect for these roles, for example behavioral, technical, or system design?",
    next_section=None,
)

"""Background section template."""

from ...enums import SectionID
from ..base_prompt import SectionTemplate

BACKGROUND_TEMPLATE = SectionTemplate(
    section_id=SectionID.BACKGROUND,
    name="Background",
    description="Understand the user's current experience, skills, and education.",
    system_prompt_template="""SECTION: Background

Goal: understand where the user is starting from.

Gather:
1. Current or most recent job title
2. Previous role titles (titles only, no employer details required)
3. Years of professional experience
4. Key skills (technical and non-technical)
5. Highest education level and field

Completion criteria: role history, skills, education, and years of experience
are all collected and the user has confirmed the summary.
""",
    # Completion checklist: each item must be covered in the confirmed summary.
    required_fields=[
        "current_role",
        "role_history",
        "years_experience",
        "skills",
        "education",
    ],
    opening_question="To start, what is your current or most recent job title?",
    next_section=SectionID.TARGET_ROLE,
)

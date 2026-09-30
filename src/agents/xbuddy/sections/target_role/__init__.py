"""Target Role section template."""

from ...enums import SectionID
from ..base_prompt import SectionTemplate

TARGET_ROLE_TEMPLATE = SectionTemplate(
    section_id=SectionID.TARGET_ROLE,
    name="Target Role",
    description="Define the roles the user is aiming for and their constraints.",
    system_prompt_template="""SECTION: Target Role

Goal: agree on what the user is aiming for.

Gather:
1. Target job titles (one to three)
2. Location as a city or region, not a full address
3. Work arrangement: remote, hybrid, onsite, or flexible
4. Top priorities in the next role (for example growth, stability, tech stack)

Completion criteria: target role, location, work type, and priorities are all
collected and the user has confirmed the summary.
""",
    # Completion checklist: each item must be covered in the confirmed summary.
    required_fields=[
        "titles",
        "location_region",
        "work_type",
        "priorities",
    ],
    opening_question="What job titles are you aiming for next? One to three is ideal.",
    next_section=SectionID.SKILL_GAP,
)

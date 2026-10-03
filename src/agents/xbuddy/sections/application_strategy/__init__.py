"""Application Strategy section template."""

from ...enums import SectionID
from ..base_prompt import SectionTemplate

APPLICATION_STRATEGY_TEMPLATE = SectionTemplate(
    section_id=SectionID.APPLICATION_STRATEGY,
    name="Application Strategy",
    description="Plan resume updates, search channels, networking, and weekly targets.",
    system_prompt_template="""SECTION: Application Strategy

Goal: turn the target and gaps into a concrete way to get interviews.

Gather:
1. Resume and LinkedIn improvements, based on the target role and gaps
2. Search channels (job boards, company sites, recruiters, referrals)
3. A networking plan (who to reach out to and how often)
4. A weekly target for applications and outreach

Completion criteria: resume actions, search channels, networking plan, and
weekly targets are defined and the user has confirmed the summary.
""",
    # Completion checklist: each item must be covered in the confirmed summary.
    required_fields=[
        "resume_actions",
        "channels",
        "networking_plan",
        "weekly_targets",
    ],
    opening_question="Based on your target role and gaps, what would you most like to improve first: your resume, your LinkedIn, or both?",
    next_section=SectionID.INTERVIEW_PREP,
)

"""Structured extraction of JobBuddyData when a section is confirmed.

Runs only on confirmation, over the section's own conversation. Output is
validated before it reaches state:
  - Requirement evidence must point at a job source the user supplied;
    evidence that doesn't is dropped, and a requirement left with no evidence
    is dropped, so every requirement stays traceable.
  - Quotes are trimmed to the model's cap. Raw postings are never stored.
Extraction failure never blocks the conversation; the caller records it.
"""

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .enums import SectionID
from .llm import get_chat_model
from .models import (
    MAX_EVIDENCE_QUOTE_CHARS,
    JobBuddyData,
    JobRequirement,
    JobSource,
    TargetRole,
    UserProfile,
)

EXTRACTION_TAG = "internal_extraction"
EXTRACTION_RETRY_TAG = "extraction_retry"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_SCHEMAS: dict[SectionID, str] = {
    SectionID.BACKGROUND: (
        '{"current_role": str|null, "role_history": [str], "years_experience": int|null, '
        '"skills": [str], "education": str|null}'
    ),
    SectionID.TARGET_ROLE: (
        '{"titles": [str], "location_region": str|null, '
        '"work_type": "remote"|"hybrid"|"onsite"|"flexible"|null, "priorities": [str]}'
    ),
    SectionID.SKILL_GAP: (
        '{"job_sources": [{"label": str, "url": str|null}], "requirements": [{"name": str, '
        '"priority": "must_have"|"nice_to_have", "evidence": [{"source_label": str, '
        f'"quote": str (max {MAX_EVIDENCE_QUOTE_CHARS} chars)}}]}}]}}'
    ),
}

EXTRACTABLE_SECTIONS = frozenset(_SCHEMAS)

EXTRACTION_PROMPT = """Extract structured data from a job-search coaching conversation.
Return ONLY a JSON object matching this shape:
{schema}

Rules:
- Use only what the user said or confirmed. Use null or [] when unknown.
- Location is a city or region only. Never include salary or contact details.
- For requirements, every evidence item must name the job source label it came
  from and quote a short phrase from that posting.
"""


def _parse_json(raw: str) -> dict[str, Any]:
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        raise ValueError("no JSON object in extraction output")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("extraction output is not a JSON object")  # noqa: TRY004
    return data


def _clean_requirements(raw: list, labels: set[str]) -> list[JobRequirement]:
    cleaned = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        evidence = [
            {**e, "quote": str(e.get("quote", ""))[:MAX_EVIDENCE_QUOTE_CHARS]}
            for e in item.get("evidence") or []
            if isinstance(e, dict) and e.get("source_label") in labels and e.get("quote")
        ]
        if evidence:
            cleaned.append(JobRequirement.model_validate({**item, "evidence": evidence}))
    return cleaned


def merge_extraction(
    section_id: SectionID, data: dict[str, Any], user_data: JobBuddyData
) -> JobBuddyData:
    """Validate extracted data and return an updated copy of user_data."""
    if section_id == SectionID.BACKGROUND:
        return user_data.model_copy(update={"profile": UserProfile.model_validate(data)})
    if section_id == SectionID.TARGET_ROLE:
        return user_data.model_copy(update={"target_role": TargetRole.model_validate(data)})
    if section_id == SectionID.SKILL_GAP:
        sources = [JobSource.model_validate(s) for s in data.get("job_sources") or []]
        requirements = _clean_requirements(data.get("requirements"), {s.label for s in sources})
        return user_data.model_copy(
            update={"job_sources": sources, "requirements": requirements}
        )
    return user_data


# Checklist items that map to structured JobBuddyData, per section.
_STRUCTURED_FIELDS: dict[SectionID, tuple[str, ...]] = {
    SectionID.BACKGROUND: ("current_role", "role_history", "years_experience", "skills", "education"),
    SectionID.TARGET_ROLE: ("titles", "location_region", "work_type", "priorities"),
    SectionID.SKILL_GAP: ("job_sources", "requirements"),
}


def structured_fields(section_id: SectionID) -> list[str]:
    return list(_STRUCTURED_FIELDS.get(section_id, ()))


def unverified_fields(section_id: SectionID, user_data: JobBuddyData) -> list[str]:
    """Structured checklist items with no extracted data behind them.

    The decision's covered_fields is the model's own claim. This compares that
    claim with what extraction actually found, so a field the model said was
    covered but that has no data is recorded instead of silently trusted.
    """
    source = {
        SectionID.BACKGROUND: user_data.profile,
        SectionID.TARGET_ROLE: user_data.target_role,
        SectionID.SKILL_GAP: user_data,
    }.get(section_id)
    if source is None:
        return []
    return [f for f in structured_fields(section_id) if getattr(source, f) in (None, "", [])]


async def extract_section_data(
    section_id: SectionID,
    conversation: list[BaseMessage],
    confirmed_summary: str,
    user_data: JobBuddyData,
    config: RunnableConfig,
    attempt: int = 1,
    max_attempts: int = 1,
) -> JobBuddyData:
    """Extract and merge structured data for a confirmed section. Raises on failure.

    Each call is named and tagged so its cost is visible in a trace: the run name
    says which section and which attempt, and retries carry the extraction_retry tag.
    """
    if section_id not in EXTRACTABLE_SECTIONS:
        return user_data
    transcript = "\n".join(
        f"{'USER' if isinstance(m, HumanMessage) else 'ASSISTANT'}: {m.content}"
        for m in conversation
    )
    prompt = [
        SystemMessage(EXTRACTION_PROMPT.format(schema=_SCHEMAS[section_id])),
        HumanMessage(f"CONFIRMED SUMMARY:\n{confirmed_summary}\n\nCONVERSATION:\n{transcript}"),
    ]
    is_retry = attempt > 1
    extract_config = {
        **config,
        "run_name": f"extract[{section_id.value}]"
        + (f" retry {attempt}/{max_attempts}" if is_retry else ""),
        "tags": [
            *(config or {}).get("tags", []),
            EXTRACTION_TAG,
            *([EXTRACTION_RETRY_TAG] if is_retry else []),
        ],
        "metadata": {
            **(config or {}).get("metadata", {}),
            "section": section_id.value,
            "extraction_attempt": attempt,
        },
    }
    response = await get_chat_model(config).ainvoke(prompt, extract_config)
    return merge_extraction(section_id, _parse_json(str(response.content)), user_data)

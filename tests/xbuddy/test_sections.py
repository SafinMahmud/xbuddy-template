"""PR 2 tests: section templates and the context loader."""

import pytest

from agents.xbuddy.enums import SectionID, SectionStatus
from agents.xbuddy.models import SectionState, default_state
from agents.xbuddy.prompts import (
    BASE_RULES,
    SECTION_TEMPLATES,
    get_next_unfinished_section,
    get_section_template,
)
from agents.xbuddy.tools import build_context_packet, get_context


def test_all_five_sections_have_templates_in_order():
    assert list(SECTION_TEMPLATES) == list(SectionID)
    for sid in SectionID:
        assert get_section_template(sid).section_id == sid
        assert get_section_template(sid.value).section_id == sid  # accepts strings


def test_next_section_chain_matches_enum_order():
    order = list(SectionID)
    for i, sid in enumerate(order):
        expected = order[i + 1] if i + 1 < len(order) else None
        assert get_section_template(sid).next_section == expected


@pytest.mark.parametrize("sid", list(SectionID))
def test_templates_have_no_placeholders_and_define_completion(sid):
    t = get_section_template(sid)
    text = f"{t.name} {t.description} {t.system_prompt_template}"
    for placeholder in ("TODO", "[TBD]", "Section 1", "..."):
        assert placeholder not in text
    assert "Completion criteria" in t.system_prompt_template
    assert t.required_fields


def test_unknown_section_fails_clearly():
    with pytest.raises(ValueError, match="Unknown section 'section_1'"):
        get_section_template("section_1")


def test_context_packet_combines_base_rules_and_section_prompt():
    packet = build_context_packet(SectionID.BACKGROUND, default_state()["section_states"])
    assert packet.section_id == SectionID.BACKGROUND
    assert packet.status == SectionStatus.PENDING
    assert BASE_RULES.strip()[:40] in packet.system_prompt
    assert "SECTION: Background" in packet.system_prompt
    assert packet.validation_rules["required_fields"] == SECTION_TEMPLATES[
        SectionID.BACKGROUND
    ].required_fields


def test_context_recaps_only_confirmed_sections():
    states = default_state()["section_states"]
    states["background"] = SectionState(
        section_id=SectionID.BACKGROUND, status=SectionStatus.DONE,
        confirmed_summary="Backend dev, 3 years, Python",
    )
    states["target_role"] = SectionState(
        section_id=SectionID.TARGET_ROLE, status=SectionStatus.IN_PROGRESS,
        confirmed_summary="unconfirmed guess",
    )
    packet = build_context_packet(SectionID.SKILL_GAP, states)

    assert "Background: Backend dev, 3 years, Python" in packet.system_prompt
    assert "unconfirmed guess" not in packet.system_prompt


def test_next_unfinished_section_can_skip_current():
    states = default_state()["section_states"]
    for sid in (SectionID.BACKGROUND, SectionID.SKILL_GAP):
        states[sid.value] = SectionState(section_id=sid, status=SectionStatus.DONE)
    assert get_next_unfinished_section(states) == SectionID.TARGET_ROLE
    assert get_next_unfinished_section(states, exclude=SectionID.TARGET_ROLE) == (
        SectionID.APPLICATION_STRATEGY
    )


@pytest.mark.asyncio
async def test_get_context_tool_accepts_serialized_state():
    states = {k: v.model_dump() for k, v in default_state()["section_states"].items()}
    result = await get_context.ainvoke(
        {"user_id": 1, "thread_id": "t", "section_id": "target_role", "section_states": states}
    )
    assert result["section_id"] == SectionID.TARGET_ROLE
    assert "SECTION: Target Role" in result["system_prompt"]

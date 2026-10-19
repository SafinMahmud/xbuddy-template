"""Generate decision node — structured routing decision after each reply.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_decision.py

The model returns JSON matching ChatAgentDecision. The output is validated in
code and never trusted blindly:
  - Malformed or invalid output falls back to "stay" and records the error.
  - "next" requires the user to be satisfied, a non-empty section summary, and
    every item of the section's completion checklist listed in covered_fields.
    A missing or partial list fails closed (stay), so the model cannot advance a
    section whose required fields were never collected.
  - "modify:<id>" must name a real section other than the current one.
  - Skill Gap cannot advance until the user has pasted a real posting.
  - When a "next" is refused, the reason is stored in `blocked_next`, and the
    next reply is told to deal with it first (see generate_reply), so the chat
    goes back to the open section instead of running ahead of the sidebar.
  - A "next" that would finish the conversation (no unfinished section left)
    is refused while the reply is still asking the user a question, so the
    roadmap never arrives before they answer.
The decision model sees the whole section (its saved draft and recent
messages), not only the last exchange, so covered_fields reflects everything
the user said in the section. Checklist names are compared loosely
("Current role" matches current_role).
The call is tagged "internal_decision" so /stream never shows it to the user.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from ..enums import RouterDirective, SectionID
from ..llm import get_chat_model
from ..models import ChatAgentDecision, ChatAgentOutput, XBuddyState
from ..prompts import get_next_unfinished_section, get_section_template

logger = logging.getLogger(__name__)

DECISION_TAG = "internal_decision"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

DECISION_PROMPT = """You decide how a job-search coaching conversation should route.
Return ONLY a JSON object, no prose, with these keys:
  "router_directive": "stay" | "next" | "modify:<section_id>"
  "is_satisfied": true | false | null   (must be true whenever router_directive is "next")
  "user_satisfaction_feedback": short string or null
  "should_save_content": true | false
  "section_summary": plain-text summary of everything collected in the current
                     section so far, or null if nothing yet
  "covered_fields": list of the completion-checklist item names (exactly as
                    written below) that the user has already provided

Rules:
- "next" ONLY if the assistant had presented a summary of the current section,
  the user explicitly confirmed it (for example "yes", "looks good"), and every
  checklist item is in covered_fields.
- covered_fields counts everything the user provided anywhere in this section's
  conversation or the saved draft, not only in the latest message.
- "modify:<section_id>" if the user wants to change an answer from an EARLIER
  section. Valid section ids: {section_ids}.
- Otherwise "stay" (still collecting, user corrected the summary, off topic).
- should_save_content is true whenever section_summary has new information.

Current section: {section_id} ({section_name})
Completion checklist: {required_fields}
"""


def _stay(reason: str | None = None) -> ChatAgentDecision:
    return ChatAgentDecision(
        router_directive=RouterDirective.STAY.value,
        user_satisfaction_feedback=reason,
    )


def parse_decision(raw: str) -> ChatAgentDecision:
    """Parse model text into a decision. Raises ValueError on malformed output."""
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        raise ValueError("no JSON object in decision output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from None
    if not isinstance(data, dict):
        # ValueError (not TypeError) so every kind of malformed output is handled alike.
        raise ValueError("decision output is not a JSON object")  # noqa: TRY004
    if isinstance(data.get("router_directive"), str):
        data["router_directive"] = data["router_directive"].strip().lower()
    try:
        return ChatAgentDecision.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid decision fields: {exc.errors()[0]['msg']}") from None


PASTED_POSTING_MIN_CHARS = 300  # a user message this long in Skill Gap counts as a posting


def section_has_posting(state: XBuddyState) -> bool:
    """True once the user has pasted a posting in this section.

    A heuristic: a user message in the section's memory at least
    PASTED_POSTING_MIN_CHARS long. It is the same text extraction reads, so a
    section can't be confirmed with nothing for extraction to quote.
    """
    # Short memory is capped, so a long Skill Gap chat could push the posting out
    # of it; the recent transcript is checked too.
    recent = [*(state.get("short_memory") or []), *(state.get("messages") or [])[-40:]]
    return any(
        isinstance(m, HumanMessage) and len(str(m.content)) >= PASTED_POSTING_MIN_CHARS
        for m in recent
    )


SECTION_HISTORY_LIMIT = 20  # section messages shown to the decision model
MESSAGE_CHARS = 600  # longer messages (pasted postings) are cut, but stay in view


def _field_key(name: str) -> str:
    """'Current role', 'current-role' and 'current_role' all become 'current_role'."""
    return "_".join(re.findall(r"[a-z0-9]+", str(name).lower()))


def missing_required_fields(decision: ChatAgentDecision, current: SectionID) -> list[str]:
    """Checklist items of the current section the decision does not report as covered."""
    covered = {_field_key(f) for f in decision.covered_fields}
    return [
        f for f in get_section_template(current).required_fields if _field_key(f) not in covered
    ]


def _clip(text: str) -> str:
    """A long message cut to MESSAGE_CHARS, saying how long it was.

    A pasted posting must stay visible to the decision for the whole section
    (it is the job source), but its full text is not needed to judge routing,
    and sending it every turn costs tokens (Groq free-tier rate limits).
    """
    if len(text) <= MESSAGE_CHARS:
        return text
    return f"{text[:MESSAGE_CHARS]} [... pasted text, {len(text)} characters in total]"


def _speaker(message) -> str:
    return "USER" if isinstance(message, HumanMessage) else "ASSISTANT"


def build_transcript(state: XBuddyState, reply: AIMessage) -> str:
    """The section so far for the decision model: saved draft, then recent messages."""
    current: SectionID = state["current_section"]
    section = (state.get("section_states") or {}).get(current.value)
    draft = getattr(getattr(section, "content", None), "plain_text", None) or "(none)"
    history = [
        m for m in (state.get("short_memory") or state.get("messages") or [])
        if isinstance(m, (HumanMessage, AIMessage)) and m is not reply
    ][-SECTION_HISTORY_LIMIT:]
    lines = [f"{_speaker(m)}: {_clip(str(m.content))}" for m in history] or ["(none)"]
    return (
        f"SAVED DRAFT OF THIS SECTION:\n{draft}\n\n"
        "SECTION CONVERSATION SO FAR:\n" + "\n\n".join(lines) + "\n\n"
        f"ASSISTANT REPLY:\n{reply.content}"
    )


# A user message that is ONLY a confirmation: "yes", "Yes, that's right!",
# "looks good, thanks". The whole message must match, so "yes, actually change
# my title" (confirmation plus a correction) never counts.
_ACK = (
    r"yes|yep|yeah|yup|correct|confirm(?:ed)?|right|exactly|perfect|sure|ok(?:ay)?|"
    r"looks good|sounds good|all good|that'?s (?:right|correct|it)|it is|go ahead|"
    r"thanks|thank you|please|great"
)
_CONFIRMATION = re.compile(
    rf"^\s*(?:{_ACK})(?:[\s,.!]+(?:{_ACK}))*[\s,.!]*$",
    re.IGNORECASE,
)
_FIRST_WORD_YES = re.compile(r"^\s*(?:yes|yep|yeah|yup|correct|confirm(?:ed)?|looks good|sounds good|ok(?:ay)?|sure)\b", re.IGNORECASE)


def is_plain_confirmation(text: str) -> bool:
    """True only when the whole message is a short acknowledgement starting with a 'yes'."""
    text = (text or "").replace("\u2019", "'")
    return bool(_FIRST_WORD_YES.match(text) and _CONFIRMATION.match(text))


def last_user_text(state: XBuddyState) -> str:
    user = next((m for m in reversed(state.get("messages") or []) if isinstance(m, HumanMessage)), None)
    return str(user.content) if user else ""


def asked_to_confirm(state: XBuddyState) -> bool:
    """True when the assistant message just before the user's reply asked a question
    (the summary's "Is this correct?"). Only then is a short 'yes' a confirmation."""
    messages = list(state.get("messages") or [])
    user_index = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), None
    )
    if not user_index:
        return False
    prior = next(
        (m for m in reversed(messages[:user_index]) if isinstance(m, AIMessage)), None
    )
    return prior is not None and "?" in str(prior.content)


def fill_satisfaction(
    decision: ChatAgentDecision, user_text: str, confirmation_asked: bool
) -> ChatAgentDecision:
    """A 'next' with is_satisfied left null counts as satisfied only when the user's
    whole message is a plain acknowledgement AND it answers a question.

    Live logs showed the model proposing 'next' with every field covered and a
    summary, but is_satisfied null, so the section stuck. "yes please" or
    "yes, thanks" are accepted because they answer "Is this correct?"; without
    that question they could be filler or a request, so they don't count. An
    explicit False is always respected.
    """
    if (
        decision.router_directive == RouterDirective.NEXT.value
        and decision.is_satisfied is None
        and confirmation_asked
        and is_plain_confirmation(user_text)
    ):
        return decision.model_copy(update={"is_satisfied": True})
    return decision


def apply_guardrails(decision: ChatAgentDecision, current: SectionID) -> ChatAgentDecision:
    """Downgrade decisions the conversation doesn't support to 'stay'."""
    directive = decision.router_directive
    if directive == RouterDirective.NEXT.value:
        incomplete = (
            decision.is_satisfied is not True
            or not (decision.section_summary or "").strip()
            or missing_required_fields(decision, current)
        )
        if incomplete:
            return decision.model_copy(update={"router_directive": RouterDirective.STAY.value})
    elif directive.startswith(f"{RouterDirective.MODIFY.value}:"):
        target = directive.split(":", 1)[1]
        if target not in {s.value for s in SectionID} or target == current.value:
            return decision.model_copy(update={"router_directive": RouterDirective.STAY.value})
    return decision


def blocked_reason(
    proposed: ChatAgentDecision, current: SectionID, state: XBuddyState, question_gate: bool
) -> str:
    """Plain words for why a proposed 'next' was refused, for the next reply."""
    reasons = []
    missing = missing_required_fields(proposed, current)
    if missing:
        reasons.append("these details are still missing: " + ", ".join(missing))
    if proposed.is_satisfied is not True:
        reasons.append("the user has not clearly confirmed the summary")
    if not (proposed.section_summary or "").strip():
        reasons.append("no summary of this section was shown yet")
    if current == SectionID.SKILL_GAP and not section_has_posting(state):
        reasons.append("no job posting has been pasted yet")
    if question_gate:
        reasons.append("the user still has to answer the last question")
    return "; ".join(reasons) or "the section is not complete yet"


async def generate_decision_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Produce a validated routing decision for the latest exchange."""
    current: SectionID = state["current_section"]
    messages = state.get("messages", [])
    reply = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None

    error_updates: dict[str, Any] = {}
    blocked: str | None = None
    if reply is None:
        decision = _stay("no reply to evaluate")
    else:
        template = get_section_template(current)
        system = DECISION_PROMPT.format(
            section_ids=", ".join(s.value for s in SectionID),
            section_id=current.value,
            section_name=template.name,
            required_fields=", ".join(template.required_fields),
        )
        transcript = build_transcript(state, reply)
        decision_config = {**config, "tags": [*(config or {}).get("tags", []), DECISION_TAG]}
        try:
            response = await get_chat_model(config).ainvoke(
                [SystemMessage(system), HumanMessage(transcript)], decision_config
            )
            proposed = fill_satisfaction(
                parse_decision(str(response.content)),
                last_user_text(state),
                asked_to_confirm(state),
            )
            decision = apply_guardrails(proposed, current)
            if (
                current == SectionID.SKILL_GAP
                and decision.router_directive == RouterDirective.NEXT.value
                and not section_has_posting(state)
            ):
                # The model may believe the section is done after a chat about
                # self-assessed gaps; without a real posting there is nothing to
                # ground the gap analysis in, so the section stays open.
                decision = decision.model_copy(
                    update={"router_directive": RouterDirective.STAY.value}
                )
            question_gate = (
                decision.router_directive == RouterDirective.NEXT.value
                and "?" in str(reply.content)
                and get_next_unfinished_section(state.get("section_states") or {}, exclude=current)
                is None
            )
            if question_gate:
                # Advancing now would write the roadmap, but the reply just asked the
                # user something. Wait for the answer.
                decision = decision.model_copy(
                    update={"router_directive": RouterDirective.STAY.value}
                )
            if decision.router_directive != proposed.router_directive:
                blocked = blocked_reason(proposed, current, state, question_gate)
                logger.info(
                    "generate_decision: %s downgraded to stay (missing fields: %s; "
                    "satisfied: %s; summary: %s; posting: %s)",
                    proposed.router_directive,
                    missing_required_fields(proposed, current) or "none",
                    proposed.is_satisfied,
                    bool((proposed.section_summary or "").strip()),
                    section_has_posting(state) if current == SectionID.SKILL_GAP else "n/a",
                )
        except Exception as exc:  # noqa: BLE001
            # Any failure (malformed output or provider error) falls back to "stay"
            # rather than crashing the turn.
            logger.warning("generate_decision fell back to stay: %s", exc)
            decision = _stay()
            error_updates = {
                "error_count": state.get("error_count", 0) + 1,
                "last_error": f"generate_decision: {exc}",
            }

    return {
        "agent_output": ChatAgentOutput(
            reply=str(reply.content) if reply else "", **decision.model_dump()
        ),
        "router_directive": decision.router_directive,
        "blocked_next": blocked,
        "awaiting_satisfaction_feedback": decision.router_directive == RouterDirective.STAY.value
        and bool(decision.section_summary),
        **error_updates,
    }

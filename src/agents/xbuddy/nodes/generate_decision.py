"""Generate decision node — structured routing decision after each reply.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_decision.py

The model returns JSON matching ChatAgentDecision. The output is validated in
code and never trusted blindly:
  - Malformed or invalid output falls back to "stay" and records the error.
  - "next" requires the user to be satisfied and a non-empty section summary.
  - "modify:<id>" must name a real section other than the current one.
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
from ..prompts import get_section_template

logger = logging.getLogger(__name__)

DECISION_TAG = "internal_decision"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

DECISION_PROMPT = """You decide how a job-search coaching conversation should route.
Return ONLY a JSON object, no prose, with these keys:
  "router_directive": "stay" | "next" | "modify:<section_id>"
  "is_satisfied": true | false | null
  "user_satisfaction_feedback": short string or null
  "should_save_content": true | false
  "section_summary": plain-text summary of everything collected in the current
                     section so far, or null if nothing yet

Rules:
- "next" ONLY if the assistant had presented a summary of the current section and
  the user explicitly confirmed it (for example "yes", "looks good").
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


def apply_guardrails(decision: ChatAgentDecision, current: SectionID) -> ChatAgentDecision:
    """Downgrade decisions the conversation doesn't support to 'stay'."""
    directive = decision.router_directive
    if directive == RouterDirective.NEXT.value:
        if decision.is_satisfied is not True or not (decision.section_summary or "").strip():
            return decision.model_copy(update={"router_directive": RouterDirective.STAY.value})
    elif directive.startswith(f"{RouterDirective.MODIFY.value}:"):
        target = directive.split(":", 1)[1]
        if target not in {s.value for s in SectionID} or target == current.value:
            return decision.model_copy(update={"router_directive": RouterDirective.STAY.value})
    return decision


async def generate_decision_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Produce a validated routing decision for the latest exchange."""
    current: SectionID = state["current_section"]
    messages = state.get("messages", [])
    reply = messages[-1] if messages and isinstance(messages[-1], AIMessage) else None

    error_updates: dict[str, Any] = {}
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
        prior_ai = next((m for m in reversed(messages[:-1]) if isinstance(m, AIMessage)), None)
        user = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        transcript = (
            f"PREVIOUS ASSISTANT MESSAGE:\n{prior_ai.content if prior_ai else '(none)'}\n\n"
            f"USER MESSAGE:\n{user.content if user else '(none)'}\n\n"
            f"ASSISTANT REPLY:\n{reply.content}"
        )
        decision_config = {**config, "tags": [*(config or {}).get("tags", []), DECISION_TAG]}
        try:
            response = await get_chat_model(config).ainvoke(
                [SystemMessage(system), HumanMessage(transcript)], decision_config
            )
            decision = apply_guardrails(parse_decision(str(response.content)), current)
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
        "awaiting_satisfaction_feedback": decision.router_directive == RouterDirective.STAY.value
        and bool(decision.section_summary),
        **error_updates,
    }

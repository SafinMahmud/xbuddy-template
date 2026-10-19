"""Generate reply node — the user-facing, streamed response.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_reply.py

  1. Uses the context packet's system prompt (base rules, section prompt,
     confirmed recap, draft, and next-section handoff).
  2. Sends the section's short memory plus the new user message. Earlier
     sections reach the model only through their confirmed summaries.
  3. Calls the model with the node's config, so LangGraph's "messages" stream
     mode forwards tokens to the /stream endpoint as they are generated.
  4. An empty model message is never sent: it is retried once, then replaced
     with the fallback reply (an empty reply reaches the browser as a cut-off turn).
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..llm import get_chat_model
from ..models import XBuddyState
from ..tools import build_context_packet

logger = logging.getLogger(__name__)

SHORT_MEMORY_LIMIT = 20  # messages kept per section
FALLBACK_REPLY = "Sorry, I had trouble responding just now. Could you say that again?"
EMPTY_REPLY_NUDGE = "\n\n---\n\nWrite your reply to the user now, in plain text."


def _pending_user_message(messages: list[BaseMessage]) -> HumanMessage | None:
    return messages[-1] if messages and isinstance(messages[-1], HumanMessage) else None


def _text(content: Any) -> str:
    """Message content as plain text (some providers return a list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else str(b.get("text", ""))
            for b in content
            if isinstance(b, (str, dict)) and (isinstance(b, str) or b.get("type") == "text")
        )
    return str(content or "")


async def generate_reply_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Generate the conversational reply for the current section."""
    packet = state.get("context_packet") or build_context_packet(
        state["current_section"], state["section_states"]
    )
    short_memory = list(state.get("short_memory") or [])
    user_msg = _pending_user_message(state.get("messages", []))
    if user_msg is not None and (not short_memory or short_memory[-1] is not user_msg):
        short_memory.append(user_msg)

    system_prompt = packet.system_prompt
    if state.get("roadmap"):
        system_prompt += (
            "\n\n---\n\nROADMAP ALREADY DELIVERED. Answer follow-up questions about it. "
            "If the user wants to change an earlier answer, acknowledge it; the roadmap "
            "will be regenerated once they confirm the change.\n\n" + state["roadmap"]
        )

    blocked = state.get("blocked_next")
    if blocked:
        system_prompt += (
            "\n\n---\n\nIMPORTANT, DO THIS FIRST: the user tried to confirm this section, "
            f"but it cannot be completed yet because {blocked}. If your last message "
            "already moved on to another section, say in one short sentence that this "
            "section needs one more thing first. Then ask only for what is missing, or "
            "show this section's summary again and ask the user to confirm it. Do not "
            "continue with any other section."
        )

    try:
        # Getting the model is inside the try: a missing key must end in the
        # fallback reply, not a failed request.
        model = get_chat_model(config)
        recent = short_memory[-SHORT_MEMORY_LIMIT:]
        response = await model.ainvoke([SystemMessage(system_prompt), *recent], config)
        content = _text(response.content)
        if not content.strip():
            # Models occasionally return an empty message. Ask once more.
            logger.warning("empty reply from the model, retrying once")
            retry = await model.ainvoke(
                [SystemMessage(system_prompt + EMPTY_REPLY_NUDGE), *recent], config
            )
            content = _text(retry.content).strip() or FALLBACK_REPLY
        reply = AIMessage(content=content, id=getattr(response, "id", None))
        error_updates: dict[str, Any] = {}
    except Exception as exc:  # network, rate limit, provider errors
        logger.exception("generate_reply failed")
        reply = AIMessage(content=FALLBACK_REPLY)
        error_updates = {
            "error_count": state.get("error_count", 0) + 1,
            "last_error": f"generate_reply: {type(exc).__name__}: {exc}",
        }

    return {
        "messages": [reply],
        "short_memory": [*short_memory, reply][-SHORT_MEMORY_LIMIT:],
        "awaiting_user_input": True,
        **error_updates,
    }

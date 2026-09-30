"""Generate reply node — the user-facing, streamed response.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_reply.py

  1. Uses the context packet's system prompt (base rules, section prompt,
     confirmed recap, draft, and next-section handoff).
  2. Sends the section's short memory plus the new user message. Earlier
     sections reach the model only through their confirmed summaries.
  3. Calls the model with the node's config, so LangGraph's "messages" stream
     mode forwards tokens to the /stream endpoint as they are generated.
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


def _pending_user_message(messages: list[BaseMessage]) -> HumanMessage | None:
    return messages[-1] if messages and isinstance(messages[-1], HumanMessage) else None


async def generate_reply_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Generate the conversational reply for the current section."""
    packet = state.get("context_packet") or build_context_packet(
        state["current_section"], state["section_states"]
    )
    short_memory = list(state.get("short_memory") or [])
    user_msg = _pending_user_message(state.get("messages", []))
    if user_msg is not None and (not short_memory or short_memory[-1] is not user_msg):
        short_memory.append(user_msg)

    prompt = [SystemMessage(packet.system_prompt), *short_memory[-SHORT_MEMORY_LIMIT:]]
    try:
        response = await get_chat_model(config).ainvoke(prompt, config)
        reply = AIMessage(content=response.content, id=getattr(response, "id", None))
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

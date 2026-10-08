"""Generate reply node — the user-facing, streamed response.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_reply.py

  1. Uses the context packet's system prompt (base rules, section prompt,
     confirmed recap, draft, and next-section handoff).
  2. Sends the section's short memory plus the new user message. Earlier
     sections reach the model only through their confirmed summaries.
  3. Calls the model with the node's config, so LangGraph's "messages" stream
     mode forwards tokens to the /stream endpoint as they are generated.
  4. Tool use: in the sections where real openings help, the model gets the
     search_jobs tool. If it asks for a search, this node stores the request in
     tool_scratch and returns without a reply; the graph runs the `tools` node
     and comes back here, where the model writes the reply from the results.
     Tool traffic never enters `messages` or short memory.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..enums import SectionID
from ..job_search import TOOLS
from ..llm import get_chat_model
from ..models import XBuddyState
from ..tools import build_context_packet

logger = logging.getLogger(__name__)

SHORT_MEMORY_LIMIT = 20  # messages kept per section
FALLBACK_REPLY = "Sorry, I had trouble responding just now. Could you say that again?"

MAX_TOOL_ROUNDS = 2  # searches the model may run in one turn before it must answer
TOOL_LIMIT_REPLY = (
    "I couldn't finish the job search just now. You can paste a posting you already "
    "have, or ask me to search again."
)
# Sections where real openings help. After the roadmap is delivered the tool stays on.
TOOL_SECTIONS = frozenset(
    {SectionID.TARGET_ROLE, SectionID.SKILL_GAP, SectionID.APPLICATION_STRATEGY}
)
JOB_SEARCH_RULES = """JOB SEARCH TOOL
- search_jobs looks up real, current job openings. Call it only when the user asks
  to see real openings, or says they have no job posting to paste.
- Build the query from the user's target titles and location unless they ask for
  something else. One search per message is normally enough.
- Show at most five results as a list: title, company, location, date posted, and
  the link, then say the listings come from the source named in the result.
- Use only what the tool returned. Never invent, complete, or reword a posting, and
  never mention salary.
- In the Skill Gap section the results are leads only: ask the user to open the ones
  they like and paste the full posting text. The gap analysis needs the real text.
- If the tool returns an error or no results, say so in one plain sentence and
  continue the section (for example, ask the user to paste a posting instead).
"""


def _pending_user_message(messages: list[BaseMessage]) -> HumanMessage | None:
    return messages[-1] if messages and isinstance(messages[-1], HumanMessage) else None


TOOL_RULES_BLOCK = "\n\n---\n\n" + JOB_SEARCH_RULES
TOOL_DOWN_BLOCK = (
    "\n\n---\n\nJob search is unavailable right now. If the user asked for job "
    "openings, say so in one sentence and continue with the current section."
)


def _tools_allowed(state: XBuddyState) -> bool:
    return state["current_section"] in TOOL_SECTIONS or bool(state.get("roadmap"))


def _with_tools(model):
    """The model with search_jobs attached, or None if it cannot call tools."""
    try:
        return model.bind_tools(TOOLS)
    except NotImplementedError:
        return None


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

    try:
        # Getting the model is inside the try: a missing key must end in the
        # fallback reply, not a failed request.
        model = get_chat_model(config)
        tool_model = _with_tools(model) if _tools_allowed(state) else None
        # This turn's tool calls and results so far (empty on the first pass).
        scratch = list(state.get("tool_scratch") or []) if tool_model is not None else []
        recent = short_memory[-SHORT_MEMORY_LIMIT:]
        if tool_model is None:
            response = await model.ainvoke([SystemMessage(system_prompt), *recent], config)
        else:
            try:
                response = await tool_model.ainvoke(
                    [SystemMessage(system_prompt + TOOL_RULES_BLOCK), *recent, *scratch], config
                )
            except Exception as exc:  # noqa: BLE001
                # Tool calling must never cost the user their reply (for example a
                # provider rejecting a malformed tool call). Answer without tools.
                logger.warning("reply with tools failed, retrying without: %s", exc)
                tool_model = None
                response = await model.ainvoke(
                    [SystemMessage(system_prompt + TOOL_DOWN_BLOCK), *recent], config
                )
        tool_calls = getattr(response, "tool_calls", None) or []
        rounds = sum(1 for m in scratch if isinstance(m, AIMessage) and m.tool_calls)
        if tool_model is not None and tool_calls and rounds < MAX_TOOL_ROUNDS:
            # Not a reply yet: hand the request to the tools node, then come back here.
            return {"tool_scratch": [*scratch, response]}
        content = response.content
        if tool_calls and not content:
            content = TOOL_LIMIT_REPLY  # still asking for tools after the limit
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
        "tool_scratch": [],
        **error_updates,
    }

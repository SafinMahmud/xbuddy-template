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

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from ..enums import SectionID
from ..job_posting import postings_block
from ..llm import get_chat_model
from ..models import XBuddyState
from ..tool_registry import TOOLS
from ..tools import build_context_packet

logger = logging.getLogger(__name__)

SHORT_MEMORY_LIMIT = 20  # messages kept per section
FALLBACK_REPLY = "Sorry, I had trouble responding just now. Could you say that again?"

MAX_TOOL_ROUNDS = 3  # tool steps the model may take in one turn before it must answer
TOOL_LIMIT_REPLY = (
    "I couldn't finish the job search just now. You can paste a posting you already "
    "have, or ask me to search again."
)
# Sections where real openings help. After the roadmap is delivered the tool stays on.
TOOL_SECTIONS = frozenset(
    {SectionID.TARGET_ROLE, SectionID.SKILL_GAP, SectionID.APPLICATION_STRATEGY}
)
JOB_SEARCH_RULES = """JOB SEARCH TOOLS
- search_jobs looks up recent job openings from company career boards. It only
  covers software roles in Ontario and remote Canada. Call it only when the user
  asks to see real openings, or says they have no job posting to paste. If they
  target another field or region, say the search does not cover it instead.
- Build the query from one to three words of the user's target title (for example
  "backend" or "python developer") and their city. If nothing is found, you may
  search once more with fewer words.
- Show at most five results as a list: title, company, location, date posted, and
  the link, then say the listings come from company career boards.
- Use only what the tool returned. Never invent, complete, or reword a posting, and
  never mention salary.
- After showing results in the Skill Gap section, ask which ones to use. The user
  can answer with numbers, names or links.
- read_job_posting reads the full text of a posting. Call it with the exact link
  from the search results when the user picks a result, or with a link the user
  pastes. When the user picks several, call it for all of them in the same step
  (several calls at once), not one per step.
- When the user picks from LAST SEARCH RESULTS below ("use 2 and 4"), use those
  links directly. Do not search again.
- When it returns a posting, use its text as the user's posting: name it by its
  label and quote short phrases from it for the requirements. Never say you read
  a posting you did not get back from the tool.
- If a tool returns an error, no results, or a posting it cannot read, give its
  message in one plain sentence and continue the section (for example, ask the
  user to paste the posting text instead).
"""


def _pending_user_message(messages: list[BaseMessage]) -> HumanMessage | None:
    return messages[-1] if messages and isinstance(messages[-1], HumanMessage) else None


TOOL_RULES_BLOCK = "\n\n---\n\n" + JOB_SEARCH_RULES
TOOL_DOWN_BLOCK = (
    "\n\n---\n\nJob search is unavailable right now. If the user asked for job "
    "openings, say so in one sentence and continue with the current section."
)


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


def _results_block(scratch: list) -> str:
    """This turn's tool results as plain text, for a retry without tools."""
    results = [m.content for m in scratch if isinstance(m, ToolMessage)]
    if not results:
        return ""
    return "\n\n---\n\nTOOL RESULTS FROM THIS TURN (use them in your reply):\n" + "\n\n".join(
        str(r) for r in results
    )


def _search_block(results: list[dict]) -> str:
    """The latest search results, numbered as shown, so a pick needs no new search."""
    if not results:
        return ""
    lines = [
        f"{i}. {r.get('title')} - {r.get('company') or 'Unknown company'}"
        f" ({r.get('location') or 'location not given'}): {r.get('url')}"
        for i, r in enumerate(results, 1)
    ]
    return "\n\n---\n\nLAST SEARCH RESULTS (numbered as the user saw them):\n" + "\n".join(lines)


EMPTY_REPLY_NUDGE = "\n\n---\n\nWrite your reply to the user now, in plain text."


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

    postings = state.get("fetched_postings") or []
    if postings and state["current_section"] == SectionID.SKILL_GAP:
        system_prompt += "\n\n---\n\n" + postings_block(postings)

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
                tool_prompt = (
                    system_prompt
                    + TOOL_RULES_BLOCK
                    + _search_block(list(state.get("last_search_results") or []))
                )
                response = await tool_model.ainvoke(
                    [SystemMessage(tool_prompt), *recent, *scratch], config
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
        content = _text(response.content)
        if not content.strip():
            # Either the model still wants tools after the limit, or it returned an
            # empty message (gpt-oss after tool results does this now and then).
            # Both used to end the turn badly, so answer once more without tools,
            # with whatever this turn's tools already returned, as text.
            reason = "tool limit reached" if tool_calls else "empty reply"
            logger.warning("%s, answering once more without tools", reason)
            retry = await model.ainvoke(
                [SystemMessage(system_prompt + _results_block(scratch) + EMPTY_REPLY_NUDGE),
                 *recent],
                config,
            )
            fallback = TOOL_LIMIT_REPLY if tool_calls else FALLBACK_REPLY
            content = _text(retry.content).strip() or fallback
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

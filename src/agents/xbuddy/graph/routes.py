"""Routing logic for XBuddy Agent graph.

These functions are used as conditional edges in the StateGraph.
Study FounderBuddy's routes.py:
https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/graph/routes.py
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage

from ..enums import RouterDirective
from ..models import XBuddyState


def route_after_memory_updater(state: XBuddyState) -> Literal["implementation", "router"]:
    """Route after memory_updater — generate final output or loop back.

    TODO: This checks should_generate_final_output. Make sure your
    memory_updater sets this flag when all sections are complete.
    """
    if state.get("should_generate_final_output", False):
        return "implementation"
    return "router"


def route_after_reply(state: XBuddyState) -> Literal["tools", "generate_decision"]:
    """After generate_reply: run the requested tool, or move on to the decision.

    generate_reply leaves an AI message with tool calls at the end of tool_scratch
    when the model asked for a tool. When it wrote the final reply, the scratch is
    empty.
    """
    scratch = state.get("tool_scratch") or []
    if scratch and isinstance(scratch[-1], AIMessage) and scratch[-1].tool_calls:
        return "tools"
    return "generate_decision"


def route_decision(state: XBuddyState) -> Literal["generate_reply"] | None:
    """Determine whether to generate a reply or end the turn.

    TODO: Study FounderBuddy's route_decision to understand the full logic
    for handling STAY/NEXT/MODIFY directives and pending user input.
    """
    def has_pending_user_input() -> bool:
        msgs = state.get("messages", [])
        if not msgs:
            return False
        return isinstance(msgs[-1], HumanMessage)

    if state.get("finished", False):
        if has_pending_user_input():
            return "generate_reply"
        return None

    directive = state.get("router_directive")

    if directive == RouterDirective.STAY or (isinstance(directive, str) and directive.lower() == "stay"):
        if has_pending_user_input():
            return "generate_reply"
        if state.get("awaiting_user_input", False):
            return None
        return None

    elif directive == RouterDirective.NEXT or (isinstance(directive, str) and directive.startswith("modify:")):
        return "generate_reply"

    return None

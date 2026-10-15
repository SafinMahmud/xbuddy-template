"""Tools node: runs the tool calls the model asked for in generate_reply.

LangGraph's prebuilt ToolNode does the same job. This one is written out because
JobBuddy keeps tool traffic in `tool_scratch` instead of `messages`, and because
the failure rules are worth seeing in one place:

  - an unknown tool name, arguments that fail the tool's schema, or a tool that
    raises all become an error result for the model, never an exception;
  - every tool call gets exactly one ToolMessage back, which is what chat model
    APIs require before the model can answer.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ..job_search import TOOLS_BY_NAME
from ..models import XBuddyState

logger = logging.getLogger(__name__)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


async def run_tools_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Execute the pending tool calls and append their results to tool_scratch."""
    scratch = list(state.get("tool_scratch") or [])
    request = scratch[-1] if scratch and isinstance(scratch[-1], AIMessage) else None
    if request is None or not request.tool_calls:
        return {}

    results = []
    for call in request.tool_calls:
        tool = TOOLS_BY_NAME.get(call["name"])
        if tool is None:
            content = _error(f"Unknown tool {call['name']!r}.")
        else:
            try:
                content = str(await tool.ainvoke(call["args"], config))
            except Exception as exc:  # noqa: BLE001 - bad arguments from the model, or a tool bug
                logger.warning("tool %s failed: %s", call["name"], exc)
                content = _error("The tool could not run with those arguments.")
        results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
    return {"tool_scratch": [*scratch, *results]}

"""Tools node: runs the tool calls the model asked for in generate_reply.

LangGraph's prebuilt ToolNode does the same job. This one is written out because
JobBuddy keeps tool traffic in `tool_scratch` instead of `messages`, and because
the failure rules are worth seeing in one place:

  - an unknown tool name, arguments that fail the tool's schema, or a tool that
    raises all become an error result for the model, never an exception;
  - every tool call gets exactly one ToolMessage back, which is what chat model
    APIs require before the model can answer.

Postings read by read_job_posting are also kept in state (fetched_postings), so
the Skill Gap section can quote them on later turns and extraction can use them.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from ..job_posting import remember_postings
from ..models import XBuddyState
from ..tool_registry import TOOLS_BY_NAME

logger = logging.getLogger(__name__)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def _search_results(result: ToolMessage) -> list[dict] | None:
    """The results of a successful search_jobs call, if this is one."""
    if result.name != "search_jobs":
        return None
    try:
        data = json.loads(result.content)
    except (TypeError, ValueError):
        return None
    results = data.get("results") if data.get("status") == "ok" else None
    return [r for r in results if isinstance(r, dict)] if isinstance(results, list) else None


def _posting(result: ToolMessage) -> dict | None:
    """The posting inside a successful read_job_posting result, if there is one."""
    if result.name != "read_job_posting":
        return None
    try:
        data = json.loads(result.content)
    except (TypeError, ValueError):
        return None
    posting = data.get("posting") if data.get("status") == "ok" else None
    return posting if isinstance(posting, dict) and posting.get("url") else None


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

    updates: dict[str, Any] = {"tool_scratch": [*scratch, *results]}
    postings = [p for p in (_posting(r) for r in results) if p]
    if postings:
        updates["fetched_postings"] = remember_postings(
            list(state.get("fetched_postings") or []), postings
        )
    searches = [r for r in (_search_results(m) for m in results) if r]
    if searches:
        updates["last_search_results"] = searches[-1]
    return updates

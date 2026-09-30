"""Implementation node — generates the final job search roadmap.

Reference: https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/nodes/generate_business_plan.py

  1. Gathers confirmed section summaries and traceable requirements.
  2. Asks the model for a Markdown roadmap and checks it against the quality
     criteria in roadmap.validate_roadmap (structure, grounding, safety).
  3. If it fails, asks for one rewrite with the failed checks as feedback. If it
     still fails, or the model errors, uses the deterministic fallback roadmap,
     which passes the checks by construction. The user never receives a roadmap
     that failed validation.
  4. Saves the roadmap to Supabase when configured (failure is recorded only).
  5. Sets finished, clears should_generate_final_output, and resets the
     directive to stay so follow-up questions are answered normally.

Generation is tagged skip_stream: tokens are not streamed, because a draft that
later fails validation must not be shown. The validated roadmap is delivered as
one message.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..enums import RouterDirective
from ..llm import get_chat_model
from ..models import XBuddyState
from ..persistence import get_roadmap_store
from ..roadmap import ROADMAP_PROMPT, build_roadmap_input, fallback_roadmap, validate_roadmap

logger = logging.getLogger(__name__)

MAX_ROADMAP_ATTEMPTS = 2  # one generation, then one rewrite with the failed checks
ROADMAP_TAG = "skip_stream"  # the service does not stream tokens with this tag


async def _generate(
    prompt_input: str, feedback: list[str] | None, config: RunnableConfig
) -> str:
    """One generation attempt. Not streamed: the user only sees a validated roadmap."""
    messages = [SystemMessage(ROADMAP_PROMPT), HumanMessage(prompt_input)]
    if feedback:
        messages.append(
            HumanMessage(
                "Your previous roadmap failed these checks:\n- "
                + "\n- ".join(feedback)
                + "\nRewrite the full roadmap and fix every one of them."
            )
        )
    gen_config = {**config, "tags": [*(config or {}).get("tags", []), ROADMAP_TAG]}
    response = await get_chat_model(config).ainvoke(messages, gen_config)
    return str(response.content).strip()


async def implementation_node(state: XBuddyState, config: RunnableConfig) -> dict[str, Any]:
    """Generate, validate, save, and deliver the roadmap."""
    section_states = state["section_states"]
    user_data = state["user_data"]
    updates: dict[str, Any] = {}
    prompt_input = build_roadmap_input(section_states, user_data)

    roadmap, problems = "", ["roadmap was not generated"]
    try:
        for _ in range(MAX_ROADMAP_ATTEMPTS):
            feedback = problems if roadmap else None
            roadmap = await _generate(prompt_input, feedback, config)
            problems = validate_roadmap(roadmap, user_data)
            if not problems:
                break
    except Exception as exc:  # noqa: BLE001 - the user must still get a roadmap
        logger.warning("roadmap generation failed: %s", exc)
        problems = [f"{type(exc).__name__}: {exc}"]

    if problems:
        # Never deliver a roadmap that fails the quality checks.
        logger.warning("using fallback roadmap; problems: %s", problems)
        roadmap = fallback_roadmap(section_states, user_data)
        updates["error_count"] = state.get("error_count", 0) + 1
        updates["last_error"] = "implementation: " + "; ".join(problems)

    store = get_roadmap_store()
    if store is not None:
        try:
            await store.save_roadmap(state["user_id"], state["thread_id"], roadmap)
        except Exception as exc:  # noqa: BLE001 - roadmap is still in the checkpoint
            logger.warning("roadmap save failed: %s", exc)
            updates["error_count"] = updates.get("error_count", state.get("error_count", 0)) + 1
            updates["last_error"] = f"persist[roadmap]: {exc}"

    return {
        **updates,
        "roadmap": roadmap,
        "messages": [AIMessage(content=roadmap)],
        "finished": True,
        "should_generate_final_output": False,
        "router_directive": RouterDirective.STAY.value,
        "context_packet": None,
    }

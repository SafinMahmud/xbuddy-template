"""Graph builder for your XBuddy Agent.

This is the core of your LangGraph agent. It wires together all the nodes
into a StateGraph with conditional edges.

Study FounderBuddy's builder.py:
https://github.com/Victoria824/FounderBuddy/blob/main/src/agents/founder_buddy/graph/builder.py
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from ..models import XBuddyState
from ..nodes import (
    implementation_node,
    generate_decision_node,
    generate_reply_node,
    initialize_node,
    memory_updater_node,
    router_node,
    run_tools_node,
)
from .routes import route_after_memory_updater, route_after_reply, route_decision


def build_xbuddy_graph():
    """Build the XBuddy agent graph.

    Graph flow:
        START -> initialize -> router -> generate_reply -> generate_decision
                    ^                      |    ^               |
                    |                      v    |               |
                    |                      tools                |
                    +------------- memory_updater <-------------+
                                        |
                                implementation -> END

    generate_reply <-> tools is the tool loop: when the model asks for a job
    search, `tools` runs it and hands the result back for the final reply.
    """
    graph = StateGraph(XBuddyState)

    # Add nodes
    graph.add_node("initialize", initialize_node)
    graph.add_node("router", router_node)
    graph.add_node("generate_reply", generate_reply_node)
    graph.add_node("tools", run_tools_node)
    graph.add_node("generate_decision", generate_decision_node)
    graph.add_node("memory_updater", memory_updater_node)
    graph.add_node("implementation", implementation_node)

    # Add edges
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "generate_reply": "generate_reply",
            None: END,
        },
    )

    graph.add_conditional_edges(
        "generate_reply",
        route_after_reply,
        {"tools": "tools", "generate_decision": "generate_decision"},
    )
    graph.add_edge("tools", "generate_reply")
    graph.add_edge("generate_decision", "memory_updater")

    graph.add_conditional_edges(
        "memory_updater",
        route_after_memory_updater,
        {
            "implementation": "implementation",
            "router": "router",
        },
    )

    graph.add_edge("implementation", END)

    # Compile with memory checkpointer
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

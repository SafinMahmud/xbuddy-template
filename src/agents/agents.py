from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from .xbuddy.agent import graph as xbuddy_agent
from schema import AgentInfo

DEFAULT_AGENT = "xbuddy"

AgentGraph = CompiledStateGraph | Pregel


@dataclass
class Agent:
    description: str
    graph: AgentGraph


agents: dict[str, Agent] = {
    "xbuddy": Agent(
        description=(
            "JobBuddy: a job-search coach that guides users through five sections and "
            "produces a personalized job search roadmap."
        ),
        graph=xbuddy_agent,
    ),
}


def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description)
        for agent_id, agent in agents.items()
    ]

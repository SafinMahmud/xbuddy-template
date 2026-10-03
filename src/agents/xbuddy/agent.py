"""XBuddy Agent — compiled graph instance."""

from .graph.builder import build_xbuddy_graph

graph = build_xbuddy_graph()


async def initialize_xbuddy_state(user_id: int | None = None) -> dict:
    """Identity and starting section for a brand-new thread.

    The service only needs the new thread_id; the graph's initialize node builds
    the full state on the first invocation.
    """
    import uuid

    from .enums import SectionID

    return {
        "user_id": user_id or 1,
        "thread_id": str(uuid.uuid4()),
        "current_section": SectionID.BACKGROUND,
    }

"""Section persistence to Supabase.

The LangGraph checkpointer is the source of truth for a thread. Supabase is a
durable copy for the frontend and other services, so a save failure is logged
and recorded but never breaks the conversation. With no Supabase credentials
configured, persistence is skipped.
"""

import asyncio
import logging
from typing import Protocol

from .models import SectionState

logger = logging.getLogger(__name__)

AGENT_ID = "jobbuddy"


class SectionStore(Protocol):
    async def save_section(self, user_id: int, thread_id: str, state: SectionState) -> None: ...


class RoadmapStore(Protocol):
    async def save_roadmap(self, user_id: int, thread_id: str, roadmap: str) -> None: ...


class SupabaseSectionStore:
    def __init__(self) -> None:
        from integrations.supabase.supabase_client import SupabaseClient

        self._client = SupabaseClient()

    async def save_section(self, user_id: int, thread_id: str, state: SectionState) -> None:
        content = state.content
        result = await asyncio.to_thread(
            self._client.save_section_state,
            user_id=user_id,
            thread_id=thread_id,
            section_id=state.section_id.value,
            content=content.content if content else {},
            plain_text=content.plain_text if content else "",
            status=state.status.value,
            satisfaction_status=state.satisfaction_status,
            confirmed_summary=state.confirmed_summary,
            agent_id=AGENT_ID,
        )
        _raise_on_failure(result)

    async def save_roadmap(self, user_id: int, thread_id: str, roadmap: str) -> None:
        # Reuses the template's business_plans table (one row per thread).
        result = await asyncio.to_thread(
            self._client.save_business_plan,
            user_id=user_id,
            thread_id=thread_id,
            content=roadmap,
            markdown_content=roadmap,
            agent_id=AGENT_ID,
        )
        _raise_on_failure(result)


def _raise_on_failure(result: dict) -> None:
    if not result.get("success"):
        raise RuntimeError(result.get("error", "unknown Supabase error"))


def _supabase_configured() -> bool:
    from core.settings import settings

    return bool(settings.SUPABASE_URL and settings.SUPABASE_SERVICE_ROLE_KEY)


def get_section_store() -> SectionStore | None:
    """Supabase store when configured, otherwise None (checkpointer only)."""
    return SupabaseSectionStore() if _supabase_configured() else None


def get_roadmap_store() -> RoadmapStore | None:
    """Supabase store when configured, otherwise None (checkpointer only)."""
    return SupabaseSectionStore() if _supabase_configured() else None

"""Supabase client for database operations."""
import logging
from typing import Optional
from supabase import create_client, Client
from core.settings import settings

logger = logging.getLogger(__name__)

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create Supabase client instance."""
    global _supabase_client
    
    if _supabase_client is None:
        supabase_url = getattr(settings, 'SUPABASE_URL', None)
        supabase_key = getattr(settings, 'SUPABASE_SERVICE_ROLE_KEY', None)
        
        logger.info(f"Supabase config check - URL: {supabase_url is not None}, Key: {supabase_key is not None}")
        
        if not supabase_url or not supabase_key:
            error_msg = (
                "Supabase credentials not configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get the actual key value (handle SecretStr)
        if hasattr(supabase_key, 'get_secret_value'):
            key_value = supabase_key.get_secret_value()
        else:
            key_value = str(supabase_key)
        
        logger.info(f"Creating Supabase client with URL: {supabase_url}")
        _supabase_client = create_client(supabase_url, key_value)
        logger.info("✅ Supabase client initialized successfully")
    
    return _supabase_client


class SupabaseClient:
    """Wrapper class for Supabase operations."""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    def save_business_plan(
        self,
        user_id: int,
        thread_id: str,
        content: str,
        markdown_content: str,
        agent_id: str = "founder-buddy"
    ) -> dict:
        """Save business plan to Supabase (synchronous operation)."""
        try:
            logger.info(f"Saving business plan to Supabase - user_id: {user_id}, thread_id: {thread_id}, content_length: {len(content)}")
            
            result = self.client.table("business_plans").upsert({
                "user_id": user_id,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "content": content,
                "markdown_content": markdown_content,
                "updated_at": "now()"
            }, on_conflict="user_id,thread_id").execute()
            
            logger.info(f"✅ Business plan saved successfully for user {user_id}, thread {thread_id}")
            logger.debug(f"Supabase response: {result.data if hasattr(result, 'data') else 'No data'}")
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"❌ Error saving business plan to Supabase: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def save_section_state(
        self,
        user_id: int,
        thread_id: str,
        section_id: str,
        content: dict,  # Tiptap JSON
        plain_text: str,
        status: str,
        satisfaction_status: Optional[str] = None,
        agent_id: str = "founder-buddy",
        confirmed_summary: Optional[str] = None,
    ) -> dict:
        """Save section state to Supabase (synchronous operation).

        Upserts on (user_id, thread_id, section_id) so repeated saves update
        the row instead of violating the unique constraint.
        """
        try:
            result = self.client.table("section_states").upsert({
                "user_id": user_id,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "section_id": section_id,
                "content": content,
                "plain_text": plain_text,
                "status": status,
                "satisfaction_status": satisfaction_status,
                "confirmed_summary": confirmed_summary,
                "updated_at": "now()"
            }, on_conflict="user_id,thread_id,section_id").execute()
            
            logger.info(f"Section state saved: {section_id} for user {user_id}")
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error saving section state: {e}")
            return {"success": False, "error": str(e)}
    
    def get_business_plan(
        self,
        user_id: int,
        thread_id: str
    ) -> Optional[dict]:
        """Get business plan from Supabase (synchronous operation)."""
        try:
            result = self.client.table("business_plans")\
                .select("*")\
                .eq("user_id", user_id)\
                .eq("thread_id", thread_id)\
                .maybe_single()\
                .execute()
            
            return result.data if result.data else None
        except Exception as e:
            logger.error(f"Error getting business plan: {e}")
            return None
    
    def get_section_states(
        self,
        user_id: int,
        thread_id: str
    ) -> list[dict]:
        """Get all section states for a thread (synchronous operation)."""
        try:
            result = self.client.table("section_states")\
                .select("*")\
                .eq("user_id", user_id)\
                .eq("thread_id", thread_id)\
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error getting section states: {e}")
            return []


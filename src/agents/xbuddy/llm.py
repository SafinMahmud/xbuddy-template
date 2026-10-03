"""Chat model access for JobBuddy nodes.

Nodes call `get_chat_model(config)` instead of `core.llm.get_model` directly, so
a request can pick a model via config["configurable"]["model"] and tests can
patch in a fake model.

Default model: settings.DEFAULT_MODEL, which is picked from whichever API key is
in .env (or set explicitly with DEFAULT_MODEL=...). core.llm.get_model() on its
own always falls back to GPT-4o, which fails without an OpenAI key.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from core.llm import get_model
from core.settings import settings


def get_chat_model(config: RunnableConfig | None = None) -> BaseChatModel:
    model_name = ((config or {}).get("configurable") or {}).get("model")
    return get_model(model_name or settings.DEFAULT_MODEL)

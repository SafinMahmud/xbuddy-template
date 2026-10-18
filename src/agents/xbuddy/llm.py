"""Chat model access for JobBuddy nodes.

Nodes call `get_chat_model(config)` instead of `core.llm.get_model` directly, so
a request can pick a model via config["configurable"]["model"] and tests can
patch in a fake model.

Default model: settings.DEFAULT_MODEL, which is picked from whichever API key is
in .env (or set explicitly with DEFAULT_MODEL=...). core.llm.get_model() on its
own always falls back to GPT-4o, which fails without an OpenAI key.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig

from core.llm import get_model
from core.settings import settings


logger = logging.getLogger(__name__)


def get_chat_model(config: RunnableConfig | None = None) -> BaseChatModel | Runnable:
    """The default model, with FALLBACK_MODEL behind it when one is set.

    A request that names its own model gets exactly that model, no fallback.
    """
    model_name = ((config or {}).get("configurable") or {}).get("model")
    if model_name:
        return get_model(model_name)
    primary = get_model(settings.DEFAULT_MODEL)
    fallback_name = settings.FALLBACK_MODEL
    if not fallback_name or fallback_name == settings.DEFAULT_MODEL:
        return primary
    try:
        fallback = get_model(fallback_name)
    except Exception as exc:  # noqa: BLE001 - a broken fallback must not break the primary
        logger.warning("fallback model %s unavailable: %s", fallback_name, exc)
        return primary
    return primary.with_fallbacks([fallback])

"""FALLBACK_MODEL: a second model answers when the default one fails."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agents.xbuddy import llm as llm_mod


class Failing(FakeListChatModel):
    async def ainvoke(self, input, config=None, **kwargs):
        raise RuntimeError("429 rate limit")


@pytest.mark.asyncio
async def test_the_fallback_answers_when_the_default_fails(monkeypatch):
    models = {"primary": Failing(responses=["x"]), "backup": FakeListChatModel(responses=["hi"])}
    monkeypatch.setattr(llm_mod, "get_model", lambda name: models[name])
    monkeypatch.setattr(llm_mod.settings, "DEFAULT_MODEL", "primary")
    monkeypatch.setattr(llm_mod.settings, "FALLBACK_MODEL", "backup")

    reply = await llm_mod.get_chat_model().ainvoke("hello")

    assert reply.content == "hi"


def test_no_fallback_configured_returns_the_default_model(monkeypatch):
    primary = FakeListChatModel(responses=["x"])
    monkeypatch.setattr(llm_mod, "get_model", lambda name: primary)
    monkeypatch.setattr(llm_mod.settings, "FALLBACK_MODEL", None)
    assert llm_mod.get_chat_model() is primary


def test_a_request_that_names_a_model_gets_no_fallback(monkeypatch):
    chosen = FakeListChatModel(responses=["x"])
    monkeypatch.setattr(llm_mod, "get_model", lambda name: chosen)
    monkeypatch.setattr(llm_mod.settings, "FALLBACK_MODEL", "backup")
    assert llm_mod.get_chat_model({"configurable": {"model": "m"}}) is chosen


def test_gemini_api_key_is_accepted_as_the_google_key(monkeypatch):
    from core.settings import Settings

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert Settings(_env_file=None).GOOGLE_API_KEY.get_secret_value() == "test-key"

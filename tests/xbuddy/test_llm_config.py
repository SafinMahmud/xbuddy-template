"""Model selection: follows the provider configured in .env (e.g. free Groq)."""

import pytest
from langchain_groq import ChatGroq

from agents.xbuddy import llm as xllm
from core.models import GroqModelName
from core.settings import Settings


def test_groq_only_env_defaults_to_llama_70b(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    s = Settings(_env_file=None)
    assert s.DEFAULT_MODEL == GroqModelName.LLAMA_33_70B


def test_groq_model_ids_match_groqcloud():
    assert GroqModelName.LLAMA_33_70B.value == "llama-3.3-70b-versatile"
    assert GroqModelName.LLAMA_31_8B.value == "llama-3.1-8b-instant"


def test_get_chat_model_uses_configured_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(xllm.settings, "DEFAULT_MODEL", GroqModelName.LLAMA_33_70B)
    model = xllm.get_chat_model({"configurable": {}})

    assert isinstance(model, ChatGroq)
    assert model.model_name == "llama-3.3-70b-versatile"
    assert model.streaming is True  # tokens reach /stream


@pytest.mark.parametrize("name", [GroqModelName.GPT_OSS_120B])
def test_request_can_override_model(monkeypatch, name):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    model = xllm.get_chat_model({"configurable": {"model": name}})
    assert model.model_name == "openai/gpt-oss-120b"

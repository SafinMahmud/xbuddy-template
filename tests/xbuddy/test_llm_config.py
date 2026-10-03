"""Model selection: follows the provider configured in .env (e.g. free Groq)."""

from langchain_groq import ChatGroq

from agents.xbuddy import llm as xllm
from core.models import GroqModelName
from core.settings import Settings


def test_groq_only_env_defaults_to_free_plan_model(monkeypatch):
    """Groq's free plan serves gpt-oss; the Llama models return model_not_found there."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    s = Settings(_env_file=None)
    assert s.DEFAULT_MODEL == GroqModelName.GPT_OSS_120B


def test_default_model_can_be_set_in_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("DEFAULT_MODEL", "openai/gpt-oss-20b")
    s = Settings(_env_file=None)
    assert s.DEFAULT_MODEL == GroqModelName.GPT_OSS_20B


def test_groq_model_ids_match_groqcloud():
    assert GroqModelName.GPT_OSS_120B.value == "openai/gpt-oss-120b"
    assert GroqModelName.GPT_OSS_20B.value == "openai/gpt-oss-20b"
    assert GroqModelName.LLAMA_33_70B.value == "llama-3.3-70b-versatile"
    assert GroqModelName.LLAMA_31_8B.value == "llama-3.1-8b-instant"


def test_get_chat_model_uses_configured_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(xllm.settings, "DEFAULT_MODEL", GroqModelName.GPT_OSS_120B)
    model = xllm.get_chat_model({"configurable": {}})

    assert isinstance(model, ChatGroq)
    assert model.model_name == "openai/gpt-oss-120b"
    assert model.streaming is True  # tokens reach /stream


def test_request_can_override_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    model = xllm.get_chat_model({"configurable": {"model": GroqModelName.GPT_OSS_20B}})
    assert model.model_name == "openai/gpt-oss-20b"

"""PR 6 deployment tests: the signed-out smoke test, auth modes, and deploy config.

The smoke test (src/smoke_test.py) is the same code that is run against the
deployed URL. Here it runs against the real FastAPI app and graph with fake
models, so no API keys or network are needed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from pydantic import SecretStr

import smoke_test
from agents.xbuddy import extraction as extraction_mod
from agents.xbuddy.agent import graph
from agents.xbuddy.nodes import generate_decision as decision_mod
from agents.xbuddy.nodes import generate_reply as reply_mod
from agents.xbuddy.nodes import implementation as impl
from agents.xbuddy.nodes import memory_updater as mu

REPO = Path(__file__).resolve().parents[2]
REPLY = "Thanks! Which industry are you in, and what mistakes do you want to avoid?"
STAY = json.dumps({"router_directive": "stay", "should_save_content": True,
                   "section_summary": "Backend dev, 3 years, Python"})
SECRET = "test-secret-123"


class ModelCalls:
    """Counts model factory calls, so tests can prove a refused request did no work."""

    def __init__(self):
        self.count = 0


@pytest.fixture
def calls(monkeypatch):
    counter = ModelCalls()

    def reply_model(config=None):
        counter.count += 1
        return GenericFakeChatModel(messages=iter([AIMessage(REPLY)]))

    def decision_model(config=None):
        counter.count += 1
        return FakeListChatModel(responses=[STAY])

    monkeypatch.setattr(reply_mod, "get_chat_model", reply_model)
    monkeypatch.setattr(decision_mod, "get_chat_model", decision_model)
    monkeypatch.setattr(extraction_mod, "get_chat_model", decision_model)
    return counter


@pytest.fixture
def client(monkeypatch, tmp_path):
    from core.settings import DatabaseType, settings

    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.SQLITE)
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(settings, "USE_SUPABASE_REALTIME", False)
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(mu, "get_section_store", lambda: None)
    monkeypatch.setattr(impl, "get_roadmap_store", lambda: None)
    original = graph.checkpointer
    from service import app

    with TestClient(app) as c:
        yield c
    graph.checkpointer = original


def protect(monkeypatch):
    from core.settings import settings

    monkeypatch.setattr(settings, "AUTH_SECRET", SecretStr(SECRET))


def failed(result):
    return [c.name for c in result.checks if not c.ok]


# --- signed out, open deployment ---------------------------------------------------


def test_signed_out_smoke_passes_on_an_open_deployment(client, calls):
    result = smoke_test.run_smoke(client, expect="open")

    assert result.mode == "open"
    assert result.passed, failed(result)
    names = [c.name for c in result.checks]
    for expected in ("invoke answers", "history has the saved turn", "stream sends the reply",
                     "stream ends with [DONE]", "roadmap is not ready after two turns"):
        assert expected in names
    assert calls.count == 4  # two turns, each one reply call and one decision call


def test_smoke_fails_when_the_model_only_returns_the_fallback(client, monkeypatch):
    def broken(config=None):
        raise RuntimeError("no API key")

    monkeypatch.setattr(reply_mod, "get_chat_model", broken)
    monkeypatch.setattr(decision_mod, "get_chat_model", broken)

    result = smoke_test.run_smoke(client)

    assert not result.passed
    assert "reply came from the model, not the fallback" in failed(result)


def test_smoke_fallback_text_matches_the_reply_node():
    assert smoke_test.FALLBACK_REPLY == reply_mod.FALLBACK_REPLY


# --- signed out, protected deployment ----------------------------------------------


def test_signed_out_smoke_on_a_protected_deployment_is_refused_everywhere(
    client, calls, monkeypatch
):
    protect(monkeypatch)

    result = smoke_test.run_smoke(client, expect="protected")

    assert result.mode == "protected"
    assert result.passed, failed(result)
    refused = [c.name for c in result.checks if c.name.endswith("refused")]
    assert len(refused) == 8  # 4 routes signed out + 4 routes with a wrong token
    assert calls.count == 0  # refused before any model call


@pytest.mark.parametrize(
    "method,path,body",
    smoke_test.PROTECTED_ROUTES,
    ids=[route[1].split("?")[0] for route in smoke_test.PROTECTED_ROUTES],
)
def test_protected_routes_return_401_without_leaking_details(
    client, calls, monkeypatch, method, path, body
):
    protect(monkeypatch)

    res = client.request(method, path, json=body) if body else client.request(method, path)

    assert res.status_code == 401
    assert res.json() == {"detail": "Unauthorized"}
    assert SECRET not in res.text
    assert calls.count == 0


def test_health_stays_public_on_a_protected_deployment(client, monkeypatch):
    protect(monkeypatch)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_signed_in_smoke_runs_the_conversation_on_a_protected_deployment(
    client, calls, monkeypatch
):
    protect(monkeypatch)

    result = smoke_test.run_smoke(client, token=SECRET, expect="protected")

    assert result.passed, failed(result)
    assert "invoke answers" in [c.name for c in result.checks]
    assert calls.count == 4


def test_wrong_token_does_not_run_the_conversation(client, calls, monkeypatch):
    protect(monkeypatch)

    result = smoke_test.run_smoke(client, token="not-the-secret")

    assert not result.passed
    assert "invoke answers" in failed(result)
    assert calls.count == 0


def test_expect_flag_catches_the_wrong_mode(client, calls, monkeypatch):
    protect(monkeypatch)

    result = smoke_test.run_smoke(client, expect="open")

    assert not result.passed
    assert failed(result) == ["deployment is open"]


# --- smoke test plumbing ------------------------------------------------------------


def test_health_wait_retries_while_the_instance_wakes_up():
    import httpx

    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(502, text="waking up")
        return httpx.Response(200, json={"status": "ok"})

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as c:
        check = smoke_test.wait_for_health(c, wait_seconds=60, sleep=lambda s: None)

    assert check.ok
    assert attempts["n"] == 3


def test_health_wait_gives_up_and_reports_the_failure():
    import httpx

    def handler(request):
        raise httpx.ConnectError("refused")

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as c:
        result = smoke_test.run_smoke(c, wait_seconds=0)

    assert not result.passed
    assert failed(result) == ["health is public"]
    report = smoke_test.format_report(result, "http://test")
    assert "RESULT: FAIL" in report


def test_report_never_prints_the_token(client, calls, monkeypatch):
    protect(monkeypatch)
    result = smoke_test.run_smoke(client, token=SECRET)
    report = smoke_test.format_report(result, "http://test")
    assert "RESULT: PASS" in report
    assert SECRET not in report


# --- streaming: replies are not dropped because of their wording -------------------


def test_stream_keeps_replies_that_mention_template_keywords(client, calls):
    # The template dropped any reply containing words such as "industry" or "mistakes".
    with client.stream("POST", "/stream", json={"message": "Backend dev", "user_id": 7}) as res:
        events = [json.loads(line[6:]) for line in res.iter_lines()
                  if line.startswith("data: ") and line != "data: [DONE]"]

    messages = [e["content"] for e in events if e["type"] == "message"]
    assert [m["content"] for m in messages if m["type"] == "ai"] == [REPLY]


# --- deployment config -------------------------------------------------------------


def test_render_blueprint_matches_the_app():
    from service import app

    blueprint = yaml.safe_load((REPO / "render.yaml").read_text())
    (service,) = blueprint["services"]

    assert service["plan"] == "free"
    assert service["runtime"] == "python"
    routes = {route.path for route in app.routes}
    assert service["healthCheckPath"] in routes
    start_script = service["startCommand"].split()[-1]
    assert (REPO / start_script).is_file()
    requirements_file = service["buildCommand"].split()[-1]
    assert (REPO / requirements_file).is_file()


def _pins() -> dict[str, list[str]]:
    """Package name -> pinned versions in requirements.txt (one per Python marker)."""
    pins: dict[str, list[str]] = {}
    for line in (REPO / "requirements.txt").read_text().splitlines():
        line = line.split(";")[0].strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"unpinned requirement: {line}"
        name, version = line.split("==")
        name = name.split("[")[0].strip().lower().replace("_", "-")
        pins.setdefault(name, []).append(version.strip())
    return pins


def test_requirements_file_is_pinned_and_covers_pyproject():
    import tomllib

    from packaging.requirements import Requirement

    pins = _pins()
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    for spec in project["dependencies"]:
        requirement = Requirement(spec)
        name = requirement.name.lower().replace("_", "-")
        assert name in pins, f"{name} is in pyproject.toml but not in requirements.txt"
        for version in pins[name]:
            if len(pins[name]) == 1:  # one pin must satisfy the pyproject range
                assert requirement.specifier.contains(version), (
                    f"requirements.txt pins {name}=={version}, pyproject wants {spec}"
                )


def test_requirements_keep_the_pins_the_service_depends_on():
    from packaging.version import Version

    pins = _pins()
    (langgraph,) = pins["langgraph"]
    (aiosqlite,) = pins["aiosqlite"]
    assert Version("0.5.4") <= Version(langgraph) < Version("0.6")  # 0.5.0 fails to import
    assert Version(aiosqlite) < Version("0.22")  # 0.22 breaks the SQLite checkpointer
    assert "pytest" not in pins  # production install only


def test_render_blueprint_commits_no_secrets():
    blueprint = yaml.safe_load((REPO / "render.yaml").read_text())
    env = {item["key"]: item for item in blueprint["services"][0]["envVars"]}

    for key in ("GROQ_API_KEY", "LANGCHAIN_API_KEY"):
        assert env[key] == {"key": key, "sync": False}  # typed into the dashboard
    for key, item in env.items():
        if any(word in key for word in ("KEY", "SECRET", "PASSWORD", "TOKEN")):
            assert "value" not in item, f"{key} must not have a committed value"


def test_startup_does_not_import_unused_provider_sdks():
    # Importing every provider SDK cost about 430 MB, too much for a 512 MB free
    # instance. Only the provider that is actually used should be imported.
    code = (
        "import sys; import service.service; "
        "heavy = [m for m in ('langchain_google_vertexai', 'langchain_aws', "
        "'langchain_google_genai', 'langchain_anthropic', 'langchain_ollama') "
        "if m in sys.modules]; print(heavy)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, cwd=REPO, check=False,
        env={"PYTHONPATH": str(REPO / "src"), "GROQ_API_KEY": "gsk_fake", "PATH": ""},
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().splitlines()[-1] == "[]"

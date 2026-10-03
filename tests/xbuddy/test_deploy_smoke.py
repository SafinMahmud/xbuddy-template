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
def start_service(monkeypatch, tmp_path):
    """Returns a context manager that starts the app. Each use is one process lifetime:
    leaving it shuts the service down, entering it again is a restart."""
    from contextlib import contextmanager

    from core.settings import DatabaseType, settings

    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.SQLITE)
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(settings, "USE_SUPABASE_REALTIME", False)
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(mu, "get_section_store", lambda: None)
    monkeypatch.setattr(impl, "get_roadmap_store", lambda: None)
    original = graph.checkpointer
    from service import app

    @contextmanager
    def start():
        with TestClient(app) as c:
            yield c

    yield start
    graph.checkpointer = original


@pytest.fixture
def client(start_service):
    with start_service() as c:
        yield c


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


# --- durability: does a thread survive a restart? ----------------------------------


def test_resume_passes_when_storage_survives_the_restart(start_service, calls):
    with start_service() as before:
        first = smoke_test.run_smoke(before)
    assert first.passed, failed(first)

    with start_service() as after:  # same database file: a restart, not a wipe
        result = smoke_test.run_smoke(after, resume=first.thread_id)

    assert result.passed, failed(result)
    names = [c.name for c in result.checks]
    assert "thread survived the restart" in names
    assert "new turn was added to the same thread" in names
    assert result.thread_id == first.thread_id


def test_resume_fails_when_storage_was_wiped(start_service, calls, tmp_path):
    with start_service() as before:
        first = smoke_test.run_smoke(before)
    for leftover in tmp_path.glob("checkpoints.db*"):  # what a temporary filesystem does
        leftover.unlink()

    with start_service() as after:
        result = smoke_test.run_smoke(after, resume=first.thread_id)

    assert not result.passed
    assert failed(result) == ["thread survived the restart"]
    (check,) = [c for c in result.checks if not c.ok]
    assert "does not survive a restart" in check.detail
    assert calls.count == 4  # the wiped thread is not silently restarted as a new one


def test_resume_on_a_protected_deployment_needs_the_token(client, calls, monkeypatch):
    protect(monkeypatch)
    result = smoke_test.run_smoke(client, resume="smoke-anything")
    assert not result.passed
    assert "resume needs --token on a protected deployment" in failed(result)


# --- the deployment describes itself ------------------------------------------------


def test_health_reports_open_access_and_non_durable_sqlite(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["access"] == "open"
    assert body["storage"] == {"type": "sqlite", "durable": False}


def test_health_reports_protected_access_and_durable_postgres(client, monkeypatch):
    from core.settings import DatabaseType, settings

    protect(monkeypatch)
    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.POSTGRES)
    body = client.get("/health").json()
    assert body["access"] == "protected"
    assert body["storage"] == {"type": "postgres", "durable": True}
    assert SECRET not in json.dumps(body)


def test_docs_page_carries_the_demo_notice_only_when_open(monkeypatch):
    from core.settings import settings
    from service.service import DEMO_NOTICE, api_description

    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    assert DEMO_NOTICE in api_description()
    assert "Do not enter personal or sensitive information" in api_description()
    protect(monkeypatch)
    assert DEMO_NOTICE not in api_description()


def test_report_shows_storage_and_thread_so_a_reader_sees_the_limit(client, calls):
    result = smoke_test.run_smoke(client)
    report = smoke_test.format_report(result, "http://test")
    assert "Storage: sqlite (NOT durable, threads are lost on restart)" in report
    assert f"Thread: {result.thread_id}" in report


# --- Postgres configuration ---------------------------------------------------------


@pytest.fixture
def pg_settings(monkeypatch):
    from core.settings import settings

    for key in ("POSTGRES_URL", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST",
                "POSTGRES_PORT", "POSTGRES_DB"):
        monkeypatch.setattr(settings, key, None)
    monkeypatch.setattr(settings, "POSTGRES_SSLMODE", "require")
    return settings


def test_postgres_url_is_used_as_given(pg_settings, monkeypatch):
    from memory.postgres import pg_manager, validate_postgres_config

    url = "postgresql://jobbuddy:pw@dpg-abc-a/jobbuddy"
    monkeypatch.setattr(pg_settings, "POSTGRES_URL", SecretStr(url))

    validate_postgres_config()  # the five parts are not needed
    assert pg_manager.get_connection_string() == url


def test_postgres_parts_build_a_url_with_ssl_and_an_escaped_password(pg_settings, monkeypatch):
    from memory.postgres import pg_manager, validate_postgres_config

    monkeypatch.setattr(pg_settings, "POSTGRES_USER", "postgres.ref")
    monkeypatch.setattr(pg_settings, "POSTGRES_PASSWORD", SecretStr("p@ss/word"))
    monkeypatch.setattr(pg_settings, "POSTGRES_HOST", "pooler.example.com")
    monkeypatch.setattr(pg_settings, "POSTGRES_PORT", 5432)
    monkeypatch.setattr(pg_settings, "POSTGRES_DB", "postgres")

    validate_postgres_config()
    assert pg_manager.get_connection_string() == (
        "postgresql://postgres.ref:p%40ss%2Fword@pooler.example.com:5432/postgres?sslmode=require"
    )


def test_postgres_config_error_names_what_is_missing_without_secrets(pg_settings, monkeypatch):
    from memory.postgres import validate_postgres_config

    monkeypatch.setattr(pg_settings, "POSTGRES_PASSWORD", SecretStr("hunter2"))
    with pytest.raises(ValueError) as error:
        validate_postgres_config()
    assert "POSTGRES_HOST" in str(error.value)
    assert "POSTGRES_URL" in str(error.value)
    assert "hunter2" not in str(error.value)


class _FakePool:
    """Stands in for the connection pool: answers SHOW server_encoding."""

    def __init__(self, encoding):
        self.encoding = encoding

    def connection(self):
        pool = self

        class _Conn:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, query):
                assert query == "SHOW server_encoding"

                class _Cursor:
                    async def fetchone(self):
                        return {"server_encoding": pool.encoding}

                return _Cursor()

        return _Conn()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoding,accepted",
    [("UTF8", True), ("SQL_ASCII", False), (b"SQL_ASCII", False)],  # the driver returns bytes there
)
async def test_postgres_startup_refuses_a_database_that_is_not_utf8(
    monkeypatch, encoding, accepted
):
    # Found with a live SQL_ASCII database: thread ids come back as bytes and every
    # turn after the first is written under a different thread id, silently lost.
    from memory.postgres import pg_manager

    monkeypatch.setattr(pg_manager, "pool", _FakePool(encoding))
    if accepted:
        await pg_manager.check_encoding()
    else:
        with pytest.raises(ValueError, match="encoding is SQL_ASCII, but UTF8 is required"):
            await pg_manager.check_encoding()


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
        if any(word in key for word in ("KEY", "SECRET", "PASSWORD", "TOKEN", "URL")):
            assert "value" not in item, f"{key} must not have a committed value"


def test_render_blueprint_wires_durable_private_postgres():
    blueprint = yaml.safe_load((REPO / "render.yaml").read_text())
    (database,) = blueprint["databases"]
    env = {item["key"]: item for item in blueprint["services"][0]["envVars"]}

    assert database["plan"] == "free"
    assert database["ipAllowList"] == []  # not reachable from the internet
    assert env["DATABASE_TYPE"]["value"] == "postgres"
    assert env["POSTGRES_URL"]["fromDatabase"] == {
        "name": database["name"],
        "property": "connectionString",
    }
    assert "SQLITE_DB_PATH" not in env


def test_render_blueprint_says_demo_only():
    assert "DEMO ONLY" in (REPO / "render.yaml").read_text()


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

"""Signed-out smoke test for a running JobBuddy API (local or deployed).

"Signed out" means the request carries no Authorization header and no cookies,
exactly what a stranger with only the URL can send.

Usage:
    uv run python src/smoke_test.py https://your-app.onrender.com
    uv run python src/smoke_test.py https://your-app.onrender.com --expect open
    uv run python src/smoke_test.py https://your-app.onrender.com --token "$AUTH_SECRET"

What it checks:
  1. /health answers without credentials (waits for a sleeping free instance).
  2. Which mode the deployment is in:
       open       no AUTH_SECRET on the server, anyone can chat
       protected  AUTH_SECRET set, every API route needs a bearer token
  3. protected: /invoke, /stream, /history and /roadmap all refuse a signed-out
     caller with 401, and a wrong token is refused too.
  4. open (or protected with --token): one real conversation turn through
     /invoke, the same thread read back through /history, a second turn through
     /stream, and /roadmap reporting "not ready". A fallback reply counts as a
     failure, because it means the model call did not work.

Exit code 0 when every check passes, 1 otherwise. No secrets are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

AGENT_ID = "xbuddy"
SMOKE_USER_ID = 990001
FIRST_MESSAGE = "I'm a backend developer with 3 years of experience, mostly Python and SQL."
SECOND_MESSAGE = "I also built a small React dashboard at my current job."
# Must match agents.xbuddy.nodes.generate_reply.FALLBACK_REPLY (checked by a test).
FALLBACK_REPLY = "Sorry, I had trouble responding just now. Could you say that again?"
PROTECTED_ROUTES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("POST", "/invoke", {"message": "hi", "user_id": SMOKE_USER_ID}),
    ("POST", "/stream", {"message": "hi", "user_id": SMOKE_USER_ID}),
    ("POST", "/history", {"thread_id": "smoke-signed-out"}),
    ("GET", f"/roadmap/{AGENT_ID}?thread_id=smoke-signed-out", None),
]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SmokeResult:
    mode: str = "unknown"
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(ok), detail))
        return bool(ok)


def _auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _request(client: httpx.Client, method: str, path: str, body: Any, token: str | None):
    kwargs: dict[str, Any] = {"headers": _auth(token)}
    if body is not None:
        kwargs["json"] = body
    return client.request(method, path, **kwargs)


def wait_for_health(client: httpx.Client, wait_seconds: float, sleep=time.sleep) -> Check:
    """Poll /health with no credentials. A sleeping free instance needs about a minute."""
    deadline = time.monotonic() + wait_seconds
    last = "no response"
    while True:
        try:
            res = client.get("/health")
            if res.status_code == 200 and res.json().get("status") == "ok":
                return Check("health is public", True, "200 ok")
            last = f"HTTP {res.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            last = type(exc).__name__
        if time.monotonic() >= deadline:
            return Check("health is public", False, last)
        sleep(5)


def _check_refused(client: httpx.Client, result: SmokeResult, token: str | None, label: str):
    for method, path, body in PROTECTED_ROUTES:
        res = _request(client, method, path, body, token)
        route = path.split("?")[0]
        result.add(f"{label}: {method} {route} refused", res.status_code == 401,
                   f"HTTP {res.status_code}")


def _check_conversation(client: httpx.Client, result: SmokeResult, token: str | None):
    thread_id = f"smoke-{uuid.uuid4()}"

    # --- /invoke: first turn on a new thread ----------------------------------
    res = _request(client, "POST", "/invoke",
                   {"message": FIRST_MESSAGE, "user_id": SMOKE_USER_ID, "thread_id": thread_id},
                   token)
    if not result.add("invoke answers", res.status_code == 200, f"HTTP {res.status_code}"):
        return
    body = res.json()
    output = body.get("output") or {}
    reply = str(output.get("content") or "")
    section = (output.get("custom_data") or {}).get("section") or {}
    result.add("invoke returns an AI reply", output.get("type") == "ai" and bool(reply.strip()),
               f"{len(reply)} chars")
    result.add("reply came from the model, not the fallback", reply.strip() != FALLBACK_REPLY,
               "fallback reply: check the model API key on the server"
               if reply.strip() == FALLBACK_REPLY else "")
    result.add("invoke keeps the thread id", body.get("thread_id") == thread_id)
    result.add("invoke reports section progress",
               section.get("id") == "background" and section.get("total_sections") == 5,
               f"section={section.get('id')} total={section.get('total_sections')}")

    # --- /history: the turn was saved -------------------------------------------
    res = _request(client, "POST", "/history", {"thread_id": thread_id}, token)
    if result.add("history answers", res.status_code == 200, f"HTTP {res.status_code}"):
        history = res.json()
        types = [m.get("type") for m in history.get("messages", [])]
        result.add("history has the saved turn", types == ["human", "ai"], f"messages={types}")
        result.add("history reports the section",
                   (history.get("section") or {}).get("id") == "background")

    # --- /stream: second turn on the same thread --------------------------------
    events: list[str] = []
    with client.stream("POST", "/stream", headers=_auth(token),
                       json={"message": SECOND_MESSAGE, "user_id": SMOKE_USER_ID,
                             "thread_id": thread_id}) as stream:
        status = stream.status_code
        if status == 200:
            events = [line[len("data: "):] for line in stream.iter_lines()
                      if line.startswith("data: ")]
    if result.add("stream answers", status == 200, f"HTTP {status}"):
        parsed = []
        for event in events:
            if event != "[DONE]":
                try:
                    parsed.append(json.loads(event))
                except ValueError:
                    pass
        kinds = {e.get("type") for e in parsed}
        streamed = "".join(str(e.get("content")) for e in parsed if e.get("type") == "token")
        result.add("stream sends the reply", bool(kinds & {"token", "message"}),
                   f"event types={sorted(k for k in kinds if k)}")
        result.add("stream ends with [DONE]", bool(events) and events[-1] == "[DONE]")
        result.add("stream reports no error", "error" not in kinds)
        result.add("stream does not leak the routing decision", "router_directive" not in streamed)

    # --- /roadmap: honest "not ready" before the five sections are done ---------
    res = _request(client, "GET", f"/roadmap/{AGENT_ID}?thread_id={thread_id}", None, token)
    if result.add("roadmap answers", res.status_code == 200, f"HTTP {res.status_code}"):
        roadmap = res.json()
        result.add("roadmap is not ready after two turns",
                   roadmap.get("success") is False and not roadmap.get("roadmap"))


def run_smoke(client: httpx.Client, token: str | None = None, expect: str = "any",
              wait_seconds: float = 0, sleep=time.sleep) -> SmokeResult:
    """Run the smoke test against `client` (an httpx.Client or FastAPI TestClient)."""
    result = SmokeResult()
    health = wait_for_health(client, wait_seconds, sleep)
    result.checks.append(health)
    if not health.ok:
        return result

    info = client.get("/info")  # signed out on purpose
    if info.status_code == 200:
        result.mode = "open"
    elif info.status_code == 401:
        result.mode = "protected"
    result.add("mode detected", result.mode != "unknown",
               f"{result.mode} (signed-out /info returned HTTP {info.status_code})")
    if result.mode == "unknown":
        return result
    if expect != "any":
        result.add(f"deployment is {expect}", result.mode == expect, f"found {result.mode}")

    if result.mode == "protected":
        _check_refused(client, result, None, "signed out")
        _check_refused(client, result, "wrong-token-" + uuid.uuid4().hex, "wrong token")
        if token:
            _check_conversation(client, result, token)
    else:
        _check_conversation(client, result, None)
    return result


def format_report(result: SmokeResult, base_url: str) -> str:
    lines = [f"Smoke test: {base_url}", f"Mode: {result.mode}"]
    for check in result.checks:
        mark = "PASS" if check.ok else "FAIL"
        lines.append(f"  [{mark}] {check.name}" + (f" ({check.detail})" if check.detail else ""))
    failed = sum(1 for c in result.checks if not c.ok)
    lines.append(f"RESULT: {'PASS' if result.passed else 'FAIL'} "
                 f"({len(result.checks) - failed}/{len(result.checks)} checks)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Signed-out smoke test for the JobBuddy API")
    parser.add_argument("base_url", help="for example https://your-app.onrender.com")
    parser.add_argument("--token", default=None,
                        help="AUTH_SECRET, to also run the conversation on a protected deployment")
    parser.add_argument("--expect", choices=["any", "open", "protected"], default="any",
                        help="fail if the deployment is not in this mode")
    parser.add_argument("--wait", type=float, default=120,
                        help="seconds to wait for /health (cold start), default 120")
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    with httpx.Client(base_url=base_url, timeout=120, follow_redirects=True) as client:
        result = run_smoke(client, token=args.token, expect=args.expect, wait_seconds=args.wait)
    print(format_report(result, base_url))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

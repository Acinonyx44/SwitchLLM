"""HTTP server for the SwitchLLM Console (stdlib only).

Each browser gets its own session (a cookie), so several people can click
through a hosted demo at once without one viewer's policy edits changing
what another sees. Simulated traffic and its replays are shared.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Callable

from .engine import ConsoleEngine

SESSION_COOKIE = "switchllm_sid"
MAX_BODY = 64 * 1024
MAX_TASK_CHARS = 8000


def page_html(static: dict[str, Any] | None = None) -> str:
    """The console page; with `static`, a self-contained offline replay."""
    html = resources.files("switchllm.console").joinpath("console.html").read_text()
    if static is None:
        return html
    blob = json.dumps(static, separators=(",", ":")).replace("</", "<\\/")
    return html.replace("<script>\n(() => {", f"<script>window.SWITCHLLM_STATIC={blob};</script>\n<script>\n(() => {{", 1)


class Sessions:
    """One ConsoleEngine per viewer, evicted when idle or when there are too many."""

    def __init__(self, factory: Callable[[], ConsoleEngine], max_sessions: int = 200, idle_seconds: int = 4 * 3600):
        self.factory, self.max, self.idle = factory, max_sessions, idle_seconds
        self._items: OrderedDict[str, tuple[ConsoleEngine, threading.Lock, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, sid: str | None) -> tuple[str, ConsoleEngine, threading.Lock]:
        now = time.monotonic()
        with self._lock:
            for key in [k for k, (_, _, seen) in self._items.items() if now - seen > self.idle]:
                del self._items[key]
            if sid and sid in self._items:
                engine, lock, _ = self._items.pop(sid)
            else:
                sid, engine, lock = secrets.token_urlsafe(18), None, threading.Lock()
            if engine is None:
                engine = self.factory()
            self._items[sid] = (engine, lock, now)
            while len(self._items) > self.max:
                self._items.popitem(last=False)
            return sid, engine, lock

    def __len__(self) -> int:
        return len(self._items)


class ConsoleServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], factory: Callable[[], ConsoleEngine], **kw: Any):
        super().__init__(address, Handler)
        self.sessions = Sessions(factory, **kw)
        self.factory = factory
        self._export: dict[str, Any] | None = None
        self._export_lock = threading.Lock()

    def export(self) -> dict[str, Any]:
        with self._export_lock:
            if self._export is None:
                self._export = self.factory().export()
            return self._export


class Handler(BaseHTTPRequestHandler):
    server: ConsoleServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter than the default
        if "/api/health" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    # -- plumbing -------------------------------------------------------------------

    def _sid(self) -> str | None:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        return cookie[SESSION_COOKIE].value if SESSION_COOKIE in cookie else None

    def _send(self, status: int, body: bytes, ctype: str, sid: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if sid:
            self.send_header("Set-Cookie", f"{SESSION_COOKIE}={sid}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any, sid: str | None = None) -> None:
        self._send(status, json.dumps(data).encode(), "application/json", sid)

    def _body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request too large")
        raw = self.rfile.read(n) if n else b"{}"
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _with_engine(self, fn: Callable[[ConsoleEngine], Any]) -> None:
        sid, engine, lock = self.server.sessions.get(self._sid())
        try:
            with lock:
                result = fn(engine)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._json(HTTPStatus.BAD_REQUEST, {"detail": str(e)}, sid)
            return
        except Exception as e:  # noqa: BLE001 -- surface, don't crash the demo
            self.log_error("error: %r", e)
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": f"{type(e).__name__}: {e}"}, sid)
            return
        self._json(HTTPStatus.OK, result, sid)

    # -- routes -----------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, page_html().encode(), "text/html; charset=utf-8")
        elif path == "/offline.html":
            self._send(HTTPStatus.OK, page_html(self.server.export()).encode(), "text/html; charset=utf-8")
        elif path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "sessions": len(self.server.sessions)})
        elif path == "/api/replay":
            self._json(HTTPStatus.OK, self.server.export())
        elif path == "/api/meta":
            self._with_engine(lambda e: e.meta())
        elif path == "/api/policy":
            self._with_engine(lambda e: e.policy_view())
        elif path == "/api/dashboard":
            self._with_engine(lambda e: e.dashboard())
        elif path == "/api/approvals":
            self._with_engine(lambda e: e.approvals_view())
        else:
            self._json(HTTPStatus.NOT_FOUND, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError) as e:
            self._json(HTTPStatus.BAD_REQUEST, {"detail": str(e)})
            return
        if path == "/api/ask":
            task = str(body.get("task", ""))[:MAX_TASK_CHARS]
            self._with_engine(lambda e: e.ask(str(body.get("persona", "")), task, bool(body.get("compare"))))
        elif path == "/api/policy/preview":
            self._with_engine(lambda e: e.preview(str(body.get("instruction", ""))[:500]))
        elif path == "/api/policy/apply":
            self._with_engine(lambda e: e.apply(str(body.get("instruction", ""))[:500]))
        elif path == "/api/feedback":
            self._with_engine(lambda e: e.feedback(str(body.get("receipt_id", "")), bool(body.get("liked")),
                                                   str(body.get("comment", ""))))
        elif path == "/api/override":
            self._with_engine(lambda e: e.request_override(str(body.get("receipt_id", "")),
                                                           str(body.get("reason", ""))))
        elif path == "/api/approvals/decide":
            self._with_engine(lambda e: e.decide(str(body.get("id", "")), bool(body.get("approve")),
                                                 str(body.get("approver") or "Manager")[:80]))
        elif path == "/api/reset":
            self._with_engine(lambda e: e.reset(body.get("ladder")))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"detail": "not found"})


def serve(host: str = "127.0.0.1", port: int = 8000, factory: Callable[[], ConsoleEngine] | None = None,
          open_browser: bool = False) -> None:
    server = ConsoleServer((host, port), factory or ConsoleEngine)
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{server.server_address[1]}/"
    print(f"SwitchLLM Console on {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.6, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

"""Minimal eihead-native monitoring Web/API.

This module intentionally uses only the Python standard library so honjia can
serve a small diagnostics surface without depending on ``apps.operator_console``.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import html
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .realtime_vision import realtime_vision_payload_from_app


JsonObject = dict[str, Any]
Clock = Callable[[], float]
ACTION_LOG_ATTRS = (
    "recent_actions",
    "action_log",
    "actions_log",
    "recent_action_log",
    "execution_log",
)


class EiheadMonitorError(RuntimeError):
    """Structured monitor error rendered as JSON."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = code
        self.details = dict(details or {})


class EiheadMonitorServer:
    """Small lifecycle wrapper around ``ThreadingHTTPServer``."""

    def __init__(self, server: ThreadingHTTPServer) -> None:
        self._server = server
        self._serving = False

    @property
    def server_address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def httpd(self) -> ThreadingHTTPServer:
        return self._server

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        self._serving = True
        try:
            self._server.serve_forever(poll_interval=poll_interval)
        finally:
            self._serving = False

    def shutdown(self) -> None:
        if self._serving:
            self._server.shutdown()

    def server_close(self) -> None:
        self._server.server_close()

    def close(self) -> None:
        self.shutdown()
        self.server_close()

    def __enter__(self) -> "EiheadMonitorServer":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class _ThreadingMonitorServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def create_handler(
    app: Any,
    *,
    clock: Clock | None = None,
    log_requests: bool = False,
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to an injectable eihead runtime app."""

    _validate_monitor_app(app)
    runtime_app = app
    now = clock or time.time

    class EiheadMonitorHandler(BaseHTTPRequestHandler):
        server_version = "eihead-monitor/0.1"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def send_error(
            self,
            code: int,
            message: str | None = None,
            explain: str | None = None,
        ) -> None:
            reason = message or _http_phrase(code)
            self._write_error(int(code), _error_code_for_status(int(code)), reason)

        def log_message(self, format: str, *args: Any) -> None:
            if log_requests:
                super().log_message(format, *args)

        def _dispatch(self, method: str) -> None:
            try:
                if method != "GET":
                    raise EiheadMonitorError(
                        HTTPStatus.METHOD_NOT_ALLOWED,
                        "method_not_allowed",
                        f"{method} is not supported by eihead monitor",
                    )
                self._route_get()
            except EiheadMonitorError as exc:
                self._write_error(exc.status_code, exc.code, str(exc), details=exc.details)
            except Exception as exc:
                self._write_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "eihead monitor request failed",
                    details={"exception": exc.__class__.__name__},
                )

        def _route_get(self) -> None:
            path = _normalize_path(self.path)
            if path == "/":
                self._write_html(HTTPStatus.OK, _render_index(runtime_app, now()))
                return
            if path == "/health":
                status_code, payload = _health_payload(runtime_app, now())
                self._write_json(status_code, payload)
                return
            if path == "/api/status":
                self._write_json(HTTPStatus.OK, _call_json_object(runtime_app, "status"))
                return
            if path == "/api/capabilities":
                self._write_json(HTTPStatus.OK, _call_json_object(runtime_app, "capabilities"))
                return
            if path in {"/api/vision/realtime", "/api/eye/realtime"}:
                self._write_json(
                    HTTPStatus.OK,
                    realtime_vision_payload_from_app(runtime_app, timestamp=now()),
                )
                return
            if path in {"/api/actions/recent", "/api/recent-actions"}:
                self._write_json(HTTPStatus.OK, _recent_actions_payload(runtime_app, now()))
                return
            raise EiheadMonitorError(HTTPStatus.NOT_FOUND, "not_found", f"unknown path: {path}")

        def _write_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(int(status_code))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _write_html(self, status_code: int, body_text: str) -> None:
            body = body_text.encode("utf-8")
            self.send_response(int(status_code))
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _write_error(
            self,
            status_code: int,
            code: str,
            message: str,
            *,
            details: Mapping[str, Any] | None = None,
        ) -> None:
            error: JsonObject = {
                "code": code,
                "message": message,
                "status_code": int(status_code),
            }
            if details:
                error["details"] = dict(details)
            self._write_json(int(status_code), {"ok": False, "error": error})

    return EiheadMonitorHandler


def create_server(
    app: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    clock: Clock | None = None,
    log_requests: bool = False,
) -> EiheadMonitorServer:
    """Create, but do not start, an eihead native monitor server."""

    handler = create_handler(app, clock=clock, log_requests=log_requests)
    return EiheadMonitorServer(_ThreadingMonitorServer((host, int(port)), handler))


def serve(
    app: Any,
    *,
    host: str = "0.0.0.0",
    port: int = 18080,
    poll_interval: float = 0.5,
    clock: Clock | None = None,
    log_requests: bool = False,
) -> None:
    """Run the native monitor until ``shutdown()`` or process termination."""

    with create_server(app, host=host, port=port, clock=clock, log_requests=log_requests) as server:
        server.serve_forever(poll_interval=poll_interval)


def _validate_monitor_app(app: Any) -> None:
    missing = [name for name in ("status", "capabilities") if not callable(getattr(app, name, None))]
    if missing:
        raise TypeError(f"eihead monitor app is missing required callables: {', '.join(missing)}")


def _normalize_path(raw_path: str) -> str:
    path = urlsplit(raw_path).path or "/"
    if path != "/":
        path = path.rstrip("/")
    return path or "/"


def _call_json_object(app: Any, method_name: str) -> JsonObject:
    method = getattr(app, method_name)
    payload = method()
    if isinstance(payload, Mapping):
        return dict(payload)
    raise EiheadMonitorError(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "invalid_runtime_payload",
        f"app.{method_name}() must return a JSON object",
        details={"payload_type": type(payload).__name__},
    )


def _health_payload(app: Any, timestamp: float) -> tuple[int, JsonObject]:
    health_fn = getattr(app, "health", None)
    if callable(health_fn):
        payload = health_fn()
        if not isinstance(payload, Mapping):
            raise EiheadMonitorError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "invalid_runtime_payload",
                "app.health() must return a JSON object",
                details={"payload_type": type(payload).__name__},
            )
        health = dict(payload)
    else:
        status_payload = _call_json_object(app, "status")
        state = str(status_payload.get("status", status_payload.get("overall_status", "ok"))).lower()
        ok = status_payload.get("ok") is not False and state not in {"error", "failed", "offline", "unhealthy"}
        health = {
            "ok": ok,
            "status": "ok" if ok else state,
            "runtime": status_payload.get("runtime", "eihead"),
            "node_id": status_payload.get("node_id", "honjia"),
            "source": "status",
            "checked_at_ts": timestamp,
        }
    return (HTTPStatus.OK if _is_healthy(health) else HTTPStatus.SERVICE_UNAVAILABLE), health


def _recent_actions_payload(app: Any, timestamp: float) -> JsonObject:
    for attr_name in ACTION_LOG_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        try:
            raw_log = source() if callable(source) else source
        except Exception as exc:
            raise EiheadMonitorError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "recent_actions_failed",
                f"failed to read action log from app.{attr_name}",
                details={"exception": exc.__class__.__name__, "source": attr_name},
            ) from exc

        actions, extra = _coerce_action_log(raw_log)
        return {
            "schema": "eihead.monitor.recent_actions.v1",
            "runtime": "eihead",
            "status": "wired",
            "wired": True,
            "source": attr_name,
            "captured_at_ts": timestamp,
            "count": len(actions),
            "actions": actions,
            **extra,
        }

    return {
        "schema": "eihead.monitor.recent_actions.v1",
        "runtime": "eihead",
        "status": "not_wired",
        "wired": False,
        "source": None,
        "captured_at_ts": timestamp,
        "count": 0,
        "actions": [],
        "message": "runtime app does not expose recent_actions or action_log",
    }


def _coerce_action_log(raw_log: Any) -> tuple[list[JsonObject], JsonObject]:
    extra: JsonObject = {}
    if raw_log is None:
        return [], extra

    if isinstance(raw_log, Mapping):
        for key in ("actions", "recent_actions", "items", "events"):
            if key in raw_log:
                actions = _coerce_action_items(raw_log[key])
                extra = {str(k): _json_ready(v) for k, v in raw_log.items() if k != key}
                return actions, extra
        return [_serialize_item(raw_log)], extra

    return _coerce_action_items(raw_log), extra


def _coerce_action_items(items: Any) -> list[JsonObject]:
    if items is None:
        return []
    if isinstance(items, (str, bytes)) or isinstance(items, Mapping):
        return [_serialize_item(items)]
    try:
        iterator = iter(items)
    except TypeError:
        return [_serialize_item(items)]
    return [_serialize_item(item) for item in iterator]


def _serialize_item(item: Any) -> JsonObject:
    if isinstance(item, Mapping):
        return {str(k): _json_ready(v) for k, v in item.items()}
    if hasattr(item, "to_dict") and callable(item.to_dict):
        payload = item.to_dict()
        if isinstance(payload, Mapping):
            return {str(k): _json_ready(v) for k, v in payload.items()}
    if is_dataclass(item):
        return {str(k): _json_ready(v) for k, v in asdict(item).items()}
    return {"value": _json_ready(item), "payload_type": type(item).__name__}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        return _json_ready(asdict(value))
    return str(value)


def _render_index(app: Any, timestamp: float) -> str:
    status = _safe_payload(lambda: _call_json_object(app, "status"))
    capabilities = _safe_payload(lambda: _call_json_object(app, "capabilities"))
    realtime = _safe_payload(lambda: realtime_vision_payload_from_app(app, timestamp=timestamp))
    recent = _safe_payload(lambda: _recent_actions_payload(app, timestamp))

    node_id = _display_value(status.get("node_id") or capabilities.get("node_id") or "honjia")
    overall = _display_value(
        status.get("overall_status")
        or status.get("status")
        or capabilities.get("overall_status")
        or "unknown"
    )
    realtime_state = _display_value(realtime.get("status", "unknown"))
    recent_state = _display_value(recent.get("status", "unknown"))

    status_json = _json_for_html(status)
    capabilities_json = _json_for_html(capabilities)
    realtime_json = _json_for_html(realtime)
    recent_json = _json_for_html(recent)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>eihead native monitor</title>
  <style>
    body {{ margin: 0; font: 15px/1.5 sans-serif; background: #f7f3ea; color: #17201a; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 28px 20px 42px; }}
    header {{ border-bottom: 3px solid #5f8f7a; margin-bottom: 20px; }}
    h1 {{ margin: 0 0 6px; font-size: 32px; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .card {{ background: #fffaf0; border: 1px solid #d7cbb3; border-radius: 14px; padding: 16px; }}
    .label {{ color: #5c6b61; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }}
    code, pre {{ background: #10231a; color: #d9f3df; border-radius: 10px; }}
    pre {{ overflow: auto; padding: 14px; }}
    a {{ color: #315f4c; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>eihead native monitor</h1>
      <p>node <strong>{node_id}</strong> · status <strong>{overall}</strong> · realtime vision <strong>{realtime_state}</strong> · actions <strong>{recent_state}</strong></p>
    </header>
    <section class="grid">
      <div class="card"><div class="label">Status API</div><a href="/api/status">/api/status</a></div>
      <div class="card"><div class="label">Capabilities API</div><a href="/api/capabilities">/api/capabilities</a></div>
      <div class="card"><div class="label">Realtime Vision API</div><a href="/api/vision/realtime">/api/vision/realtime</a></div>
      <div class="card"><div class="label">Recent Actions API</div><a href="/api/actions/recent">/api/actions/recent</a></div>
      <div class="card"><div class="label">Health API</div><a href="/health">/health</a></div>
    </section>
    <h2>Status</h2>
    <pre>{status_json}</pre>
    <h2>Capabilities</h2>
    <pre>{capabilities_json}</pre>
    <h2>Realtime Vision</h2>
    <pre>{realtime_json}</pre>
    <h2>Recent Actions</h2>
    <pre>{recent_json}</pre>
  </main>
</body>
</html>
"""


def _safe_payload(factory: Callable[[], JsonObject]) -> JsonObject:
    try:
        return factory()
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "error": {
                "code": "render_failed",
                "message": str(exc),
                "exception": exc.__class__.__name__,
            },
        }


def _json_for_html(payload: Mapping[str, Any]) -> str:
    return html.escape(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True))


def _display_value(value: Any) -> str:
    return html.escape(str(value))


def _is_healthy(payload: Mapping[str, Any]) -> bool:
    state = str(payload.get("status", "ok")).lower()
    if payload.get("ok") is False:
        return False
    return state not in {"error", "failed", "offline", "unhealthy"}


def _http_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "HTTP error"


def _error_code_for_status(status_code: int) -> str:
    if status_code == HTTPStatus.NOT_FOUND:
        return "not_found"
    if status_code == HTTPStatus.METHOD_NOT_ALLOWED:
        return "method_not_allowed"
    if status_code >= 500:
        return "internal_error"
    return "http_error"


__all__ = [
    "ACTION_LOG_ATTRS",
    "EiheadMonitorError",
    "EiheadMonitorServer",
    "create_handler",
    "create_server",
    "serve",
]

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

from .neck import build_neck_diagnostics_from_app
from .realtime_vision import realtime_vision_payload_from_app
from .voice import build_voice_diagnostics_from_app


JsonObject = dict[str, Any]
Clock = Callable[[], float]
ACTION_LOG_ATTRS = (
    "recent_actions",
    "action_log",
    "actions_log",
    "recent_action_log",
    "execution_log",
)
EVENT_LOG_ATTRS = (
    "recent_events",
)
EVENT_PAYLOAD_KEYS = frozenset(
    {
        "schema",
        "runtime",
        "status",
        "wired",
        "source",
        "captured_at_ts",
        "count",
        "events",
        "recent_events",
        "items",
        "actions",
    }
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
            if path in {"/api/voice/realtime", "/api/audio/realtime"}:
                self._write_json(
                    HTTPStatus.OK,
                    build_voice_diagnostics_from_app(runtime_app, timestamp=now()),
                )
                return
            if path in {"/api/neck/status", "/api/neck/realtime"}:
                self._write_json(
                    HTTPStatus.OK,
                    build_neck_diagnostics_from_app(runtime_app, timestamp=now()),
                )
                return
            if path in {"/api/actions/recent", "/api/recent-actions"}:
                self._write_json(HTTPStatus.OK, _recent_actions_payload(runtime_app, now()))
                return
            if path in {"/api/events/recent", "/api/recent-events"}:
                self._write_json(HTTPStatus.OK, _recent_events_payload(runtime_app, now()))
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


def _recent_events_payload(app: Any, timestamp: float) -> JsonObject:
    for attr_name in EVENT_LOG_ATTRS:
        if not hasattr(app, attr_name):
            continue
        source = getattr(app, attr_name)
        try:
            raw_log = source() if callable(source) else source
        except Exception as exc:
            raise EiheadMonitorError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "recent_events_failed",
                f"failed to read event log from app.{attr_name}",
                details={"exception": exc.__class__.__name__, "source": attr_name},
            ) from exc

        events, extra = _coerce_event_log(raw_log)
        return {
            "schema": "eihead.monitor.recent_events.v1",
            "runtime": "eihead",
            "status": "wired",
            "wired": True,
            "source": attr_name,
            "captured_at_ts": timestamp,
            "count": len(events),
            "events": events,
            **extra,
        }

    event_journal = getattr(app, "event_journal", None)
    recent = getattr(event_journal, "recent", None)
    if callable(recent):
        try:
            raw_log = recent()
        except Exception as exc:
            raise EiheadMonitorError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "recent_events_failed",
                "failed to read event log from app.event_journal.recent",
                details={"exception": exc.__class__.__name__, "source": "event_journal.recent"},
            ) from exc

        events, extra = _coerce_event_log(raw_log)
        return {
            "schema": "eihead.monitor.recent_events.v1",
            "runtime": "eihead",
            "status": "wired",
            "wired": True,
            "source": "event_journal.recent",
            "captured_at_ts": timestamp,
            "count": len(events),
            "events": events,
            **extra,
        }

    return {
        "schema": "eihead.monitor.recent_events.v1",
        "runtime": "eihead",
        "status": "not_wired",
        "wired": False,
        "source": None,
        "captured_at_ts": timestamp,
        "count": 0,
        "events": [],
        "message": "runtime app does not expose recent_events or event_journal.recent",
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


def _coerce_event_log(raw_log: Any) -> tuple[list[JsonObject], JsonObject]:
    extra: JsonObject = {}
    if raw_log is None:
        return [], extra

    if isinstance(raw_log, Mapping):
        for key in ("events", "recent_events", "items", "actions"):
            if key in raw_log:
                events = _coerce_action_items(raw_log[key])
                extra = {str(k): _json_ready(v) for k, v in raw_log.items() if str(k) not in EVENT_PAYLOAD_KEYS}
                return events, extra
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
    voice = _safe_payload(lambda: build_voice_diagnostics_from_app(app, timestamp=timestamp))
    neck = _safe_payload(lambda: build_neck_diagnostics_from_app(app, timestamp=timestamp))
    recent = _safe_payload(lambda: _recent_actions_payload(app, timestamp))
    recent_events = _safe_payload(lambda: _recent_events_payload(app, timestamp))

    node_id = _display_value(status.get("node_id") or capabilities.get("node_id") or "honjia")
    overall = _display_value(
        status.get("overall_status")
        or status.get("status")
        or capabilities.get("overall_status")
        or "unknown"
    )
    realtime_state = _display_value(realtime.get("status", "unknown"))
    voice_state = _display_value(voice.get("status", "unknown"))
    neck_state = _display_value(neck.get("status", "unknown"))
    recent_state = _display_value(recent.get("status", "unknown"))
    recent_events_state = _display_value(recent_events.get("status", "unknown"))
    realtime_diagnostic = realtime.get("diagnostic") if isinstance(realtime.get("diagnostic"), Mapping) else {}
    vision_status = _display_value(realtime_diagnostic.get("status") or realtime.get("status", "unknown"))
    vision_fps = _display_value(_metric_value(realtime_diagnostic.get("fps")))
    vision_top_detection = _display_value(_top_detection_summary(realtime_diagnostic.get("top_detection")))
    vision_frame_age = _display_value(_metric_value(realtime_diagnostic.get("last_frame_age"), suffix="s"))
    vision_backend = _display_value(realtime_diagnostic.get("backend") or "unknown")
    vision_frame_interval = _display_value(_metric_value(realtime.get("frame_interval_ms"), suffix="ms"))
    vision_jitter_guard = _display_value(realtime.get("jitter_guard") if realtime.get("jitter_guard") is not None else "unknown")
    vision_top_k = _display_value(_metric_value(realtime.get("top_k")))
    vision_score_threshold = _display_value(_metric_value(realtime.get("score_threshold")))
    vision_hooks_used = _display_value(_hooks_used_summary(realtime.get("hooks_used")))
    vision_pipeline = _display_value(_pipeline_summary(realtime.get("pipeline")))
    vision_devices = _display_value(_devices_summary(realtime.get("devices")))
    vision_readiness = _display_value(realtime.get("readiness_message") or "unknown")
    vision_parse_errors = _display_value(_metric_value(realtime.get("parse_error_count")))
    voice_ear = _display_value(_voice_component_summary(voice.get("ear"), kind="ear"))
    voice_mouth = _display_value(_voice_component_summary(voice.get("mouth"), kind="mouth"))
    voice_dialogue = _display_value(_voice_dialogue_summary(voice.get("dialogue")))
    voice_latency = _display_value(_metric_value(_voice_latency_total_ms(voice.get("latency")), suffix="ms"))
    voice_bottleneck = _display_value(_voice_bottleneck_summary(voice.get("bottleneck")))
    voice_last_turn = _display_value(_voice_last_turn_summary(voice.get("last_turn")))
    voice_round = _display_value(_voice_round_summary(voice.get("round")))
    voice_scheduler = _display_value(_voice_scheduler_summary(voice.get("scheduler")))
    voice_fast_think = _display_value(_voice_realtime_component_summary(voice.get("fast_think")))
    voice_slow_reasoner = _display_value(_voice_realtime_component_summary(voice.get("slow_reasoner")))
    voice_arbiter = _display_value(_voice_realtime_component_summary(voice.get("arbiter")))
    voice_speech_action_plan = _display_value(_voice_realtime_component_summary(voice.get("speech_action_plan")))
    voice_proactive_activity = _display_value(_voice_realtime_component_summary(voice.get("proactive_activity")))
    voice_interruption = _display_value(_voice_interruption_summary(voice.get("interruption")))
    voice_microfeedback = _display_value(_voice_microfeedback_summary(voice.get("microfeedback")))
    voice_closed_loop = _display_value(_voice_closed_loop_summary(voice.get("closed_loop_state")))
    voice_event_count = _display_value(_metric_value(voice.get("event_count")))
    voice_last_reply_delta = _display_value(voice.get("last_reply_delta") or "unknown")
    voice_first_reply_token = _display_value(
        _metric_value(_voice_latency_stage_ms(voice.get("latency"), "first_reply_token"), suffix="ms")
    )
    voice_first_speech = _display_value(
        _metric_value(_voice_latency_stage_ms(voice.get("latency"), "first_speech"), suffix="ms")
    )
    voice_cancellation_chain = _display_value(_voice_cancellation_chain_summary(voice.get("cancellation_chain")))
    voice_readiness = _display_value(voice.get("readiness_message") or "unknown")
    neck_current_angle = _display_value(_metric_value(neck.get("current_angle"), suffix="deg"))
    neck_target_angle = _display_value(_metric_value(neck.get("target_angle"), suffix="deg"))
    neck_will_move = _display_value(neck.get("will_move") if neck.get("will_move") is not None else "unknown")
    neck_suppressed = _display_value(neck.get("suppressed") if neck.get("suppressed") is not None else "unknown")
    neck_suppression_reason = _display_value(neck.get("suppression_reason") or "unknown")
    neck_servo = _display_value(_neck_servo_summary(neck.get("servo")))
    neck_axis_support = _display_value(_neck_axis_support_summary(neck.get("axis_support")))
    neck_readiness = _display_value(neck.get("readiness_message") or "unknown")

    status_json = _json_for_html(status)
    capabilities_json = _json_for_html(capabilities)
    realtime_json = _json_for_html(realtime)
    voice_json = _json_for_html(voice)
    neck_json = _json_for_html(neck)
    recent_json = _json_for_html(recent)
    recent_events_json = _json_for_html(recent_events)

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
    .metric {{ display: block; margin-top: 4px; font-size: 22px; font-weight: 700; }}
    code, pre {{ background: #10231a; color: #d9f3df; border-radius: 10px; }}
    code {{ padding: 1px 5px; }}
    pre {{ overflow: auto; padding: 14px; }}
    a {{ color: #315f4c; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>eihead native monitor</h1>
      <p>node <strong>{node_id}</strong> · status <strong>{overall}</strong> · realtime vision <strong>{realtime_state}</strong> · voice <strong>{voice_state}</strong> · neck <strong>{neck_state}</strong> · actions <strong>{recent_state}</strong> · events <strong>{recent_events_state}</strong></p>
    </header>
    <section class="grid">
      <div class="card"><div class="label">Status API</div><a href="/api/status">/api/status</a></div>
      <div class="card"><div class="label">Capabilities API</div><a href="/api/capabilities">/api/capabilities</a></div>
      <div class="card"><div class="label">Realtime Vision API</div><a href="/api/vision/realtime">/api/vision/realtime</a></div>
      <div class="card"><div class="label">Voice Diagnostics API</div><a href="/api/voice/realtime">/api/voice/realtime</a></div>
      <div class="card"><div class="label">Neck Diagnostics API</div><a href="/api/neck/status">/api/neck/status</a></div>
      <div class="card"><div class="label">Recent Actions API</div><a href="/api/actions/recent">/api/actions/recent</a></div>
      <div class="card"><div class="label">Recent Events API</div><a href="/api/events/recent">/api/events/recent</a></div>
      <div class="card"><div class="label">Health API</div><a href="/health">/health</a></div>
    </section>
    <h2>Realtime Vision Diagnostic</h2>
    <section class="grid">
      <div class="card"><div class="label">Status</div><span class="metric">{vision_status}</span></div>
      <div class="card"><div class="label">FPS</div><span class="metric">{vision_fps}</span></div>
      <div class="card"><div class="label">Top detection</div><span class="metric">{vision_top_detection}</span></div>
      <div class="card"><div class="label">Frame age</div><span class="metric">{vision_frame_age}</span></div>
      <div class="card"><div class="label">Backend</div><span class="metric">{vision_backend}</span></div>
      <div class="card"><div class="label">Frame interval</div><span class="metric">{vision_frame_interval}</span></div>
      <div class="card"><div class="label">Jitter guard</div><span class="metric">{vision_jitter_guard}</span></div>
      <div class="card"><div class="label">Top K</div><span class="metric">{vision_top_k}</span></div>
      <div class="card"><div class="label">Score threshold</div><span class="metric">{vision_score_threshold}</span></div>
      <div class="card"><div class="label">Hooks used</div><span class="metric">{vision_hooks_used}</span></div>
      <div class="card"><div class="label">Pipeline</div><span class="metric">{vision_pipeline}</span></div>
      <div class="card"><div class="label">Devices</div><span class="metric">{vision_devices}</span></div>
      <div class="card"><div class="label">Readiness</div><span class="metric">{vision_readiness}</span></div>
      <div class="card"><div class="label">Parse errors</div><span class="metric">{vision_parse_errors}</span></div>
    </section>
    <p>Realtime JSON below includes <code>boxes</code> and <code>scores</code> for direct visual diagnostics.</p>
    <h2>Neck Diagnostics</h2>
    <section class="grid">
      <div class="card"><div class="label">Status</div><span class="metric">{neck_state}</span></div>
      <div class="card"><div class="label">Current angle</div><span class="metric">{neck_current_angle}</span></div>
      <div class="card"><div class="label">Target angle</div><span class="metric">{neck_target_angle}</span></div>
      <div class="card"><div class="label">Will move</div><span class="metric">{neck_will_move}</span></div>
      <div class="card"><div class="label">Suppressed</div><span class="metric">{neck_suppressed}</span></div>
      <div class="card"><div class="label">Suppression reason</div><span class="metric">{neck_suppression_reason}</span></div>
      <div class="card"><div class="label">Servo</div><span class="metric">{neck_servo}</span></div>
      <div class="card"><div class="label">Axis support</div><span class="metric">{neck_axis_support}</span></div>
      <div class="card"><div class="label">Readiness</div><span class="metric">{neck_readiness}</span></div>
    </section>
    <h2>Voice Diagnostics</h2>
    <section class="grid">
      <div class="card"><div class="label">Status</div><span class="metric">{voice_state}</span></div>
      <div class="card"><div class="label">Ear</div><span class="metric">{voice_ear}</span></div>
      <div class="card"><div class="label">Mouth</div><span class="metric">{voice_mouth}</span></div>
      <div class="card"><div class="label">Dialogue</div><span class="metric">{voice_dialogue}</span></div>
      <div class="card"><div class="label">Latency</div><span class="metric">{voice_latency}</span></div>
      <div class="card"><div class="label">Bottleneck</div><span class="metric">{voice_bottleneck}</span></div>
      <div class="card"><div class="label">Last turn</div><span class="metric">{voice_last_turn}</span></div>
      <div class="card"><div class="label">Round</div><span class="metric">{voice_round}</span></div>
      <div class="card"><div class="label">Scheduler</div><span class="metric">{voice_scheduler}</span></div>
      <div class="card"><div class="label">Fast think</div><span class="metric">{voice_fast_think}</span></div>
      <div class="card"><div class="label">Slow reasoner</div><span class="metric">{voice_slow_reasoner}</span></div>
      <div class="card"><div class="label">Arbiter</div><span class="metric">{voice_arbiter}</span></div>
      <div class="card"><div class="label">Speech/action plan</div><span class="metric">{voice_speech_action_plan}</span></div>
      <div class="card"><div class="label">Proactive activity</div><span class="metric">{voice_proactive_activity}</span></div>
      <div class="card"><div class="label">Interrupts</div><span class="metric">{voice_interruption}</span></div>
      <div class="card"><div class="label">Microfeedback</div><span class="metric">{voice_microfeedback}</span></div>
      <div class="card"><div class="label">Closed loop</div><span class="metric">{voice_closed_loop}</span></div>
      <div class="card"><div class="label">Realtime events</div><span class="metric">{voice_event_count}</span></div>
      <div class="card"><div class="label">Last reply delta</div><span class="metric">{voice_last_reply_delta}</span></div>
      <div class="card"><div class="label">First reply token</div><span class="metric">{voice_first_reply_token}</span></div>
      <div class="card"><div class="label">First speech</div><span class="metric">{voice_first_speech}</span></div>
      <div class="card"><div class="label">Cancellation chain</div><span class="metric">{voice_cancellation_chain}</span></div>
      <div class="card"><div class="label">Readiness</div><span class="metric">{voice_readiness}</span></div>
    </section>
    <h2>Status</h2>
    <pre>{status_json}</pre>
    <h2>Capabilities</h2>
    <pre>{capabilities_json}</pre>
    <h2>Realtime Vision</h2>
    <pre>{realtime_json}</pre>
    <h2>Voice</h2>
    <pre>{voice_json}</pre>
    <h2>Neck</h2>
    <pre>{neck_json}</pre>
    <h2>Recent Actions</h2>
    <pre>{recent_json}</pre>
    <h2>Recent Events</h2>
    <pre>{recent_events_json}</pre>
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


def _metric_value(value: Any, *, suffix: str = "") -> str:
    if value in (None, ""):
        return "unknown"
    return f"{value}{suffix}"


def _top_detection_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "none"
    label = value.get("label", "unknown")
    score = value.get("score", value.get("confidence"))
    if score in (None, ""):
        return str(label)
    return f"{label} ({score})"


def _hooks_used_summary(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return ", ".join(_metric_value(item) for item in value)
    return _metric_value(value)


def _pipeline_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    backend = value.get("backend") or value.get("transport") or value.get("source")
    sink = value.get("sink")
    if backend and sink:
        return f"{backend} -> {sink}"
    if backend:
        return _metric_value(backend)
    return _metric_value(value)


def _devices_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    camera = value.get("camera") or value.get("camera_device")
    hailo = value.get("hailo") or value.get("hailo_device")
    if camera and hailo:
        return f"{camera}, {hailo}"
    return _metric_value(value)


def _neck_servo_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    status = value.get("status") or "unknown"
    available = value.get("available")
    reason = value.get("reason")
    parts = [str(status)]
    if available is not None:
        parts.append("available" if available is True else "unavailable")
    if reason and reason != "unknown":
        parts.append(str(reason))
    return " / ".join(parts)


def _neck_axis_support_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    parts: list[str] = []
    for axis in ("pan", "tilt"):
        axis_payload = value.get(axis)
        if not isinstance(axis_payload, Mapping):
            parts.append(f"{axis}=unknown")
            continue
        status = axis_payload.get("status")
        supported = axis_payload.get("supported")
        reason = axis_payload.get("reason")
        if status:
            rendered = str(status)
        elif supported is True:
            rendered = "supported"
        elif supported is False:
            rendered = "unsupported"
        else:
            rendered = "unknown"
        if reason:
            rendered = f"{rendered} ({reason})"
        parts.append(f"{axis}={rendered}")
    return " / ".join(parts)


def _voice_component_summary(value: Any, *, kind: str) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    state = value.get("state") or value.get("status") or "unknown"
    if kind == "ear":
        provider = value.get("provider")
        if provider:
            return f"{state} ({provider})"
        return str(state)
    backend = value.get("backend")
    model = value.get("model")
    if backend and model:
        return f"{state} ({backend}/{model})"
    if backend:
        return f"{state} ({backend})"
    return str(state)


def _voice_dialogue_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    phase = value.get("phase") or value.get("last_status")
    transcript = value.get("last_transcript")
    if phase and transcript:
        return f"{phase}: {transcript}"
    if phase:
        return str(phase)
    return "unknown"


def _voice_latency_total_ms(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    return value.get("total_ms")


def _voice_latency_stage_ms(value: Any, key: str) -> Any:
    if not isinstance(value, Mapping):
        return None
    stage_latency = value.get("stage_latency_ms")
    if isinstance(stage_latency, Mapping):
        return stage_latency.get(key)
    return None


def _voice_bottleneck_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    stage = value.get("stage")
    latency_ms = value.get("latency_ms")
    if stage and latency_ms not in (None, ""):
        return f"{stage} ({latency_ms}ms)"
    if stage:
        return str(stage)
    return "unknown"


def _voice_last_turn_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    transcript = value.get("transcript") or value.get("text")
    reply = value.get("reply") or value.get("response")
    if transcript and reply:
        return f"{transcript} -> {reply}"
    if transcript:
        return str(transcript)
    if reply:
        return str(reply)
    return "unknown"


def _voice_round_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    round_id = value.get("current_round_id") or value.get("round_id")
    token = value.get("current_cancellation_token") or value.get("cancellation_token")
    if not round_id:
        return "unknown"
    if isinstance(token, Mapping) and token.get("cancelled") is True:
        return f"{round_id} (cancelled)"
    if token:
        return f"{round_id} (cancel token)"
    return str(round_id)


def _voice_scheduler_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    state = value.get("state") or value.get("status") or value.get("component_state") or "unknown"
    active_round = value.get("active_round_id") or value.get("round_id")
    stale = value.get("stale") is True
    parts = [str(state)]
    if active_round:
        parts.append(str(active_round))
    if stale and "stale" not in {part.lower() for part in parts}:
        parts.append("stale")
    return " / ".join(parts)


def _voice_realtime_component_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    summary = value.get("summary")
    if summary not in (None, ""):
        return str(summary)
    state = value.get("state") or value.get("status") or value.get("component_state") or "unknown"
    return str(state)


def _voice_interruption_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    state = value.get("state") or "unknown"
    parts = [str(state)]
    interrupt_count = value.get("interrupt_count")
    interrupted_round_count = value.get("interrupted_round_count")
    if interrupt_count is not None:
        parts.append(f"{interrupt_count} interrupts")
    if interrupted_round_count is not None:
        parts.append(f"{interrupted_round_count} rounds")
    last_interrupt = value.get("last_interrupt")
    if isinstance(last_interrupt, Mapping):
        reason = last_interrupt.get("reason") or last_interrupt.get("type")
        if reason:
            parts.append(str(reason))
    if value.get("stale") is True and "stale" not in {part.lower() for part in parts}:
        parts.append("stale")
    return " / ".join(parts)


def _voice_microfeedback_summary(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, Mapping):
        label = value.get("text") or value.get("last") or value.get("label") or value.get("status") or value.get("message")
        score = value.get("score")
        if label and score not in (None, ""):
            return f"{label} ({score})"
        if label:
            return str(label)
    return _metric_value(value)


def _voice_closed_loop_summary(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    parts: list[str] = []
    for key in ("final_asr", "first_reply_delta", "first_speech", "complete"):
        if key in value:
            parts.append(f"{key}={'yes' if value.get(key) is True else 'no'}")
    return " / ".join(parts) if parts else "unknown"


def _voice_cancellation_chain_summary(value: Any) -> str:
    if value is None:
        return "none"
    if not isinstance(value, (list, tuple)):
        return _metric_value(value)
    if not value:
        return "none"
    targets: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            target = item.get("target") or item.get("event_type") or item.get("reason")
            if target:
                targets.append(str(target))
        elif item not in (None, ""):
            targets.append(str(item))
    if not targets:
        return "none"
    return " -> ".join(targets)


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

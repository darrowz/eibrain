"""Lightweight monitoring web for honjia runtime diagnostics."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from apps.operator_console.app import OperatorConsoleApp


class MonitoringWebServer:
    def __init__(self, *, runtime, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.runtime = runtime
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.console = OperatorConsoleApp()

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                report = outer.console.build_status_report(
                    body_snapshot=outer.runtime.snapshot(),
                    cognitive_snapshot={},
                    traces=outer.runtime.recent_events(),
                )
                if self.path in {"/status.json", "/healthz"}:
                    body = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
                    status_code = 200
                    if self.path == "/healthz" and report.get("system_health") != "healthy":
                        status_code = 503
                    self.send_response(status_code)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = outer._render_html(report).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        self._server = None

    @staticmethod
    def _render_html(report: dict[str, object]) -> str:
        body = report.get("body", {})
        recent = report.get("recent_traces", [])
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>eibrain honjia monitor</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f5f7fb; color: #172033; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
    pre {{ white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>honjia monitoring</h1>
  <div class="grid">
    <section class="card"><h2>system</h2><pre>{json.dumps(report, ensure_ascii=False, indent=2)}</pre></section>
    <section class="card"><h2>body</h2><pre>{json.dumps(body, ensure_ascii=False, indent=2)}</pre></section>
    <section class="card" id="recent-events"><h2>recent-events</h2><pre>{json.dumps(recent, ensure_ascii=False, indent=2)}</pre></section>
  </div>
</body>
</html>"""

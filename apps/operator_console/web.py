"""Lightweight monitoring web for honjia runtime diagnostics."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlparse

from apps.operator_console.app import OperatorConsoleApp


class MonitoringWebServer:
    def __init__(self, *, runtime, cognitive_runtime=None, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.runtime = runtime
        self.cognitive_runtime = cognitive_runtime
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.console = OperatorConsoleApp()

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
                body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                request_path = urlparse(self.path).path
                report = outer.console.build_status_report(
                    body_snapshot=outer.runtime.snapshot(),
                    cognitive_snapshot=outer.cognitive_runtime.snapshot() if outer.cognitive_runtime is not None else {},
                    traces=outer.runtime.recent_events(),
                )
                if request_path == "/vision/latest.jpg":
                    frame_path = outer.runtime.latest_visual_frame_path()
                    if frame_path and Path(frame_path).exists():
                        body = Path(frame_path).read_bytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self.send_error(404, "vision frame not available")
                    return
                if request_path in {"/status.json", "/healthz", "/metrics.json"}:
                    self._send_json(200, report)
                    return
                body = outer._render_html(report).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                request_path = urlparse(self.path).path
                if request_path != "/identity/register":
                    self.send_error(404, "not found")
                    return
                content_length = int(self.headers.get("Content-Length", "0") or 0)
                payload: dict[str, object] = {}
                if content_length:
                    try:
                        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        self._send_json(400, {"ok": False, "error": f"invalid json: {exc}"})
                        return
                if not isinstance(payload, dict):
                    self._send_json(400, {"ok": False, "error": "json body must be an object"})
                    return
                register = getattr(outer.runtime, "register_current_identity", None)
                if register is None:
                    self._send_json(501, {"ok": False, "error": "identity registration is unavailable"})
                    return
                result = register(
                    display_name=str(payload.get("display_name", "Darrow") or "Darrow"),
                    actor_id=str(payload.get("actor_id", "darrow") or "darrow"),
                )
                self._send_json(200 if result.get("ok") else 409, result)

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
        initial_report = json.dumps(report, ensure_ascii=False)
        template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>eibrain honjia monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #151718;
      --panel: #1d1f20;
      --panel-2: #232628;
      --panel-3: #2a2d30;
      --text: #f6f4ef;
      --muted: #a9acae;
      --orange: #f9a03f;
      --orange-soft: rgba(249, 160, 63, 0.18);
      --green: #29c17e;
      --red: #ff5c69;
      --yellow: #ffd166;
      --border: rgba(255,255,255,0.08);
      --shadow: 0 24px 80px rgba(0,0,0,0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(249,160,63,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(255,255,255,0.06), transparent 24%),
        linear-gradient(180deg, #111315 0%, #151718 50%, #121415 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 22px;
    }}
    .hero-side {{
      display: grid;
      grid-template-columns: repeat(2, minmax(220px, 1fr));
      gap: 16px;
      width: min(100%, 520px);
    }}
    .headline h1 {{
      margin: 0;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .headline p {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.5;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 14px;
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--text);
      font-size: 13px;
      font-weight: 600;
    }}
    .pill .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--green);
      box-shadow: 0 0 14px rgba(41, 193, 126, 0.65);
    }}
    .pill.degraded .dot {{ background: var(--red); box-shadow: 0 0 14px rgba(255, 92, 105, 0.6); }}
    .summary-grid, .organ-grid, .detail-grid {{
      display: grid;
      gap: 16px;
    }}
    .summary-grid {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 16px;
    }}
    .organ-grid {{
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      margin-bottom: 16px;
    }}
    .detail-grid {{
      grid-template-columns: 1.4fr 1fr;
    }}
    .vision-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(300px, 0.7fr);
      gap: 16px;
      align-items: start;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(35,38,40,0.96), rgba(29,31,32,0.96));
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .card h2, .card h3 {{
      margin: 0 0 10px;
      letter-spacing: -0.02em;
    }}
    .metric-value {{
      font-size: 30px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .mini-card {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .organ-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .subfunction-list, .timeline, .warning-list {{
      display: grid;
      gap: 10px;
    }}
    .subfunction-item, .timeline-item {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
    }}
    .sub-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 6px;
    }}
    .driver-tag, .health-tag, .warning-tag {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 700;
      border: 1px solid var(--border);
      background: var(--panel-3);
    }}
    .health-tag.healthy {{ color: var(--green); }}
    .health-tag.waiting {{ color: var(--yellow); background: rgba(255, 209, 102, 0.10); }}
    .health-tag.degraded, .warning-tag {{ color: var(--yellow); background: rgba(255, 209, 102, 0.12); }}
    .health-tag.unavailable {{ color: var(--red); background: rgba(255, 92, 105, 0.12); }}
    .latency-bar {{
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
      margin-top: 8px;
    }}
    .latency-bar span {{
      display: block;
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--orange), #ffcd70);
    }}
    .vision-stage {{
      position: relative;
      border-radius: 18px;
      overflow: hidden;
      background: #0d0f10;
      border: 1px solid var(--border);
      min-height: 260px;
    }}
    .vision-stage img {{
      display: block;
      width: 100%;
      height: auto;
      object-fit: cover;
    }}
    .vision-empty {{
      min-height: 260px;
      display: grid;
      place-items: center;
      color: var(--muted);
      padding: 24px;
      text-align: center;
    }}
    .bbox {{
      position: absolute;
      border: 2px solid var(--orange);
      background: rgba(249, 160, 63, 0.08);
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.2);
    }}
    .bbox.face {{
      border-color: var(--green);
      background: rgba(41, 193, 126, 0.10);
    }}
    .bbox-label {{
      position: absolute;
      top: -24px;
      left: 0;
      background: rgba(13, 15, 16, 0.92);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .action-row {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .action-button {{
      border: 1px solid rgba(249,160,63,0.5);
      border-radius: 999px;
      background: var(--orange-soft);
      color: var(--text);
      cursor: pointer;
      font-weight: 800;
      padding: 9px 13px;
    }}
    .action-button:hover {{ background: rgba(249,160,63,0.28); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 10px 8px;
      text-align: left;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .muted {{ color: var(--muted); }}
    details {{
      margin-top: 14px;
      background: var(--panel-2);
      border-radius: 14px;
      padding: 12px;
      border: 1px solid var(--border);
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      color: #f5eee5;
      font-size: 12px;
      margin: 8px 0 0;
    }}
    @media (max-width: 960px) {{
      .hero {{ flex-direction: column; }}
      .hero-side {{ grid-template-columns: 1fr; width: 100%; }}
      .detail-grid {{ grid-template-columns: 1fr; }}
      .vision-layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="headline">
        <div class="pill" id="system-pill"><span class="dot"></span><span id="system-pill-text">Loading…</span></div>
        <h1>honjia organ observability</h1>
        <p>PostHog-style live diagnostics for embodied runtime health, response latency, organ capability shifts, and failure triage.</p>
      </div>
      <div class="hero-side">
        <div class="card">
          <div class="muted">Avg latency</div>
          <div class="metric-value" id="hero-avg-latency">—</div>
          <div class="metric-label">Cross-module average heartbeat latency</div>
        </div>
        <div class="card">
          <div class="muted">Refresh cadence</div>
          <div class="metric-value">2s</div>
          <div class="metric-label" id="refresh-meta">Polling /metrics.json without WebSocket overhead</div>
        </div>
      </div>
    </section>

    <section class="summary-grid" id="summary-grid"></section>
    <section class="organ-grid" id="organ-grid"></section>
    <section class="detail-grid">
      <section class="card">
        <h2>Latency leaderboard</h2>
        <table>
          <thead><tr><th>Module</th><th>Driver</th><th>Health</th><th>Latency</th></tr></thead>
          <tbody id="latency-table"></tbody>
        </table>
      </section>
      <section class="card">
        <h2>Runtime posture</h2>
        <div class="mini-grid" id="runtime-overview"></div>
        <div class="mini-grid" id="capability-grid" style="margin-top: 14px;"></div>
        <div class="mini-grid" id="driver-breakdown" style="margin-top: 14px;"></div>
        <h3 style="margin-top: 18px;">Warnings & degradation</h3>
        <div class="warning-list" id="warning-list"></div>
        <div class="mini-grid" id="event-breakdown"></div>
      </section>
    </section>

    <section class="card" style="margin-top: 16px;">
      <h2>Audio diagnostics</h2>
      <div class="mini-grid" id="audio-summary"></div>
      <div class="subfunction-list" id="audio-events" style="margin-top: 14px;"></div>
    </section>

    <section class="card" style="margin-top: 16px;">
      <h2>Dialogue loop</h2>
      <div class="mini-grid" id="dialogue-summary"></div>
      <div class="subfunction-list" id="dialogue-events" style="margin-top: 14px;"></div>
    </section>

    <section class="card" style="margin-top: 16px;">
      <h2>Visual diagnostics</h2>
      <div class="vision-layout">
        <div class="vision-stage" id="vision-stage"></div>
        <div>
          <div class="action-row">
            <button class="action-button" type="button" onclick="registerIdentity()">Register Darrow</button>
            <span class="muted" id="identity-action-status">Session identity not registered</span>
          </div>
          <div class="mini-grid" id="vision-summary"></div>
          <div class="subfunction-list" id="vision-detection-list" style="margin-top: 14px;"></div>
        </div>
      </div>
    </section>

    <section class="card" style="margin-top: 16px;">
      <h2>Hardware probes</h2>
      <table>
        <thead><tr><th>Probe</th><th>Status</th><th>Device</th><th>Readiness</th><th>Latency</th></tr></thead>
        <tbody id="probe-table"></tbody>
      </table>
    </section>

    <section class="card" style="margin-top: 16px;">
      <h2>Recent events</h2>
      <div class="timeline" id="recent-events"></div>
      <details>
        <summary>Raw JSON</summary>
        <pre id="raw-json"></pre>
      </details>
    </section>
  </div>
  <script>
    const INITIAL_REPORT = __INITIAL_REPORT__;
    const refreshMs = 2000;

    function healthClass(value) {{
      if (value === 'healthy' || value === 'normal' || value === 'live' || value === 'played' || value === 'planned') return 'healthy';
      if (value === 'waiting' || value === 'waiting_for_data' || value === 'waiting_for_frame' || value === 'waiting_for_action' || value === 'waiting_for_target') return 'waiting';
      if (value === 'unavailable' || value === 'error') return 'unavailable';
      return 'degraded';
    }}

    function fmtLatency(value) {{
      return typeof value === 'number' ? `${{value.toFixed(2)}} ms` : '—';
    }}

    function fmtSeconds(value) {{
      return typeof value === 'number' ? `${{value.toFixed(2)}} s` : '—';
    }}

    function fmtBool(value) {{
      if (value === true) return 'present';
      if (value === false) return 'missing';
      return 'n/a';
    }}

    function fmtTime(value) {{
      if (!value) return '—';
      const date = new Date(value * 1000);
      return date.toLocaleTimeString();
    }}

    function renderSummary(report) {{
      const summary = report.summary || {{}};
      document.getElementById('hero-avg-latency').textContent = fmtLatency(summary.avg_latency_ms);
      const cards = [
        ['Healthy modules', `${{summary.healthy_subfunction_count ?? 0}} / ${{summary.subfunction_count ?? 0}}`, 'Healthy subfunctions across all organs'],
        ['Live data', `${{summary.live_data_subfunction_count ?? 0}} / ${{summary.subfunction_count ?? 0}}`, 'Subfunctions with fresh runtime output'],
        ['Enabled capabilities', `${{summary.enabled_capability_count ?? 0}} / ${{summary.capability_count ?? 0}}`, 'Current embodied capability coverage'],
        ['Warning count', String(summary.warning_count ?? 0), 'Active diagnostic warnings'],
        ['Degraded organs', String(summary.degraded_organ_count ?? 0), 'Organs needing attention'],
        ['Real drivers', String(summary.real_driver_count ?? 0), 'Non-noop driver probes in the live graph'],
        ['Unavailable probes', String(summary.unavailable_probe_count ?? 0), 'Hardware probes that are currently missing'],
      ];
      document.getElementById('summary-grid').innerHTML = cards.map(([label, value, hint]) => `
        <article class="card">
          <div class="muted">${{label}}</div>
          <div class="metric-value">${{value}}</div>
          <div class="metric-label">${{hint}}</div>
        </article>
      `).join('');
    }}

    function renderOrgans(report) {{
      const cards = report.organ_cards || [];
      document.getElementById('organ-grid').innerHTML = cards.map((card) => `
        <article class="card">
          <div class="organ-header">
            <div>
              <div class="muted">${{card.name}}</div>
              <h2>${{card.label}}</h2>
            </div>
            <span class="health-tag ${{healthClass(card.data_health || card.health)}}">${{card.data_status || card.health}}</span>
          </div>
          <div class="mini-grid">
            <div class="mini-card"><div class="muted">Healthy subfunctions</div><div class="metric-value" style="font-size:22px;">${{card.healthy_subfunctions}}</div></div>
            <div class="mini-card"><div class="muted">Live data</div><div class="metric-value" style="font-size:22px;">${{card.live_data_subfunctions ?? 0}} / ${{card.subfunction_count ?? 0}}</div></div>
            <div class="mini-card"><div class="muted">Avg latency</div><div class="metric-value" style="font-size:22px;">${{fmtLatency(card.avg_latency_ms)}}</div></div>
            <div class="mini-card"><div class="muted">Degraded subfunctions</div><div class="metric-value" style="font-size:22px;">${{card.degraded_subfunctions}}</div></div>
            <div class="mini-card"><div class="muted">Max latency</div><div class="metric-value" style="font-size:22px;">${{fmtLatency(card.max_latency_ms)}}</div></div>
          </div>
          <div class="subfunction-list" style="margin-top: 14px;">
            ${(card.subfunctions || []).map((sub) => `
              <div class="subfunction-item">
                <div class="sub-top">
                  <strong>${{sub.name}}</strong>
                  <span class="health-tag ${{healthClass(sub.data_health || sub.health)}}">${{sub.data_status || sub.health}}</span>
                </div>
                <div class="muted">Driver <span class="driver-tag">${{sub.driver}}</span> · Health ${{sub.health || '—'}} · Status ${{sub.status || '—'}}</div>
                <div class="metric-label">${{sub.visual_summary || (sub.probe?.device ? `device=${{sub.probe.device}} · ${{fmtBool(sub.probe.device_exists)}}` : (sub.error || 'No active error'))}}</div>
                <div class="latency-bar"><span style="width:${{Math.min(100, (sub.elapsed_ms || 0) / 2)}}%"></span></div>
                <div class="metric-label">${{fmtLatency(sub.elapsed_ms)}}</div>
              </div>
            `).join('')}
          </div>
        </article>
      `).join('');
    }}

    function renderRuntime(report) {{
      const runtime = report.runtime_overview || {{}};
      const capabilities = report.capability_status || [];
      const drivers = report.driver_breakdown || [];

      document.getElementById('runtime-overview').innerHTML = [
        ['Node', runtime.node_id || 'unknown'],
        ['Degradation', runtime.degradation_mode || 'unknown'],
        ['Organs', String(runtime.organ_count ?? 0)],
        ['Recent events', String(runtime.recent_event_count ?? 0)],
      ].map(([label, value]) => `<div class="mini-card"><div class="muted">${{label}}</div><div class="metric-value" style="font-size:22px;">${{value}}</div></div>`).join('');

      document.getElementById('capability-grid').innerHTML = capabilities.length
        ? capabilities.map((capability) => `
            <div class="mini-card">
              <div class="muted">${{capability.name}}</div>
              <div class="metric-value" style="font-size:20px;">${{capability.enabled ? 'on' : 'off'}}</div>
            </div>
          `).join('')
        : '<div class="muted">No capability data</div>';

      document.getElementById('driver-breakdown').innerHTML = drivers.length
        ? drivers.map((driver) => `
            <div class="mini-card">
              <div class="muted">driver:${{driver.driver}}</div>
              <div class="metric-value" style="font-size:20px;">${{driver.count}}</div>
            </div>
          `).join('')
        : '<div class="muted">No driver breakdown</div>';
    }}

    function renderLatencies(report) {{
      const rows = (report.latency_metrics || []).slice(0, 8).map((metric) => `
        <tr>
          <td><strong>${{metric.id}}</strong></td>
          <td>${{metric.driver}}</td>
          <td><span class="health-tag ${{healthClass(metric.health)}}">${{metric.health}}</span></td>
          <td>${{fmtLatency(metric.elapsed_ms)}}</td>
        </tr>
      `).join('');
      document.getElementById('latency-table').innerHTML = rows || '<tr><td colspan="4" class="muted">No latency metrics yet</td></tr>';
    }}

    function renderWarnings(report) {{
      const warnings = [...(report.warnings || [])];
      if ((report.degraded_organs || []).length) {{
        warnings.push(`degraded_organs=${{report.degraded_organs.join(', ')}}`);
      }}
      document.getElementById('warning-list').innerHTML = warnings.length
        ? warnings.map((warning) => `<div class="warning-tag">${{warning}}</div>`).join('')
        : '<div class="muted">No active warnings</div>';

      const events = report.event_breakdown || [];
      document.getElementById('event-breakdown').innerHTML = events.length
        ? events.slice(0, 4).map((event) => `<div class="mini-card"><div class="muted">${{event.kind}}</div><div class="metric-value" style="font-size:22px;">${{event.count}}</div></div>`).join('')
        : '<div class="muted">No recent event breakdown</div>';
    }}

    function renderVision(report) {{
      const visual = report.visual_diagnostics || {{}};
      const timestamp = visual.frame_captured_at_ts || report.generated_at_ts || Date.now() / 1000;
      const detections = visual.detections || [];
      const identityCandidates = visual.identity_candidates || [];
      const registeredIdentity = visual.registered_identity || {{}};
      const recognizedIdentity = visual.recognized_identity || {{}};
      const identityName = recognizedIdentity.display_name || registeredIdentity.display_name || '';
      document.getElementById('identity-action-status').textContent = registeredIdentity.registered
        ? `Registered: ${{identityName || 'known person'}}`
        : 'Session identity not registered';
      document.getElementById('vision-summary').innerHTML = [
        ['Frame', visual.frame_available ? 'live' : 'missing'],
        ['Frame age', fmtSeconds(visual.frame_age_s)],
        ['Data', visual.data_status || 'unknown'],
        ['Detection', visual.detection_status || visual.detection_health || 'unknown'],
        ['Tracking', visual.tracking_status || 'idle'],
        ['Identity', identityName || (visual.identity_status || 'unknown')],
        ['Targets', String(visual.detection_count ?? 0)],
      ].map(([label, value]) => `<div class="mini-card"><div class="muted">${{label}}</div><div class="metric-value" style="font-size:20px;">${{value}}</div></div>`).join('');

      if (visual.frame_url) {{
        const boxes = detections.map((detection) => {{
          const bbox = detection.bbox || {{}};
          const left = Math.max(0, Math.min(100, (bbox.x_min || 0) * 100));
          const top = Math.max(0, Math.min(100, (bbox.y_min || 0) * 100));
          const width = Math.max(1, Math.min(100, ((bbox.x_max || 0) - (bbox.x_min || 0)) * 100));
          const height = Math.max(1, Math.min(100, ((bbox.y_max || 0) - (bbox.y_min || 0)) * 100));
          const boxClass = detection.label === 'face' ? 'bbox face' : 'bbox';
          const label = detection.identity || detection.label || 'target';
          return `<div class="${{boxClass}}" style="left:${{left}}%;top:${{top}}%;width:${{width}}%;height:${{height}}%;"><span class="bbox-label">${{label}} ${{Number(detection.score || 0).toFixed(2)}}</span></div>`;
        }}).join('');
        document.getElementById('vision-stage').innerHTML = `
          <img src="${{visual.frame_url}}?t=${{timestamp}}" alt="latest honjia frame">
          ${{boxes}}
        `;
      }} else {{
        document.getElementById('vision-stage').innerHTML = `
          <div class="vision-empty">
            <div>
              <strong>Visual frame unavailable</strong>
              <div class="metric-label">${{visual.scene_summary || 'waiting for camera pipeline to produce a frame'}}</div>
            </div>
          </div>
        `;
      }}

      const listItems = [];
      if (visual.scene_summary) {{
        listItems.push(`<div class="subfunction-item"><div class="sub-top"><strong>Scene</strong><span class="health-tag ${{healthClass(visual.detection_health || 'unknown')}}">${{visual.detection_status || visual.detection_health || 'unknown'}}</span></div><div class="metric-label">${{visual.scene_summary}}</div></div>`);
      }}
      if (visual.identity_summary) {{
        listItems.push(`<div class="subfunction-item"><div class="sub-top"><strong>Identity</strong><span class="health-tag ${{healthClass(visual.identity_health || 'unknown')}}">${{visual.identity_status || visual.identity_health || 'unknown'}}</span></div><div class="metric-label">${{visual.identity_summary}}</div></div>`);
      }}
      if (visual.tracking_running || visual.tracking_target || visual.tracking_last_error) {{
        const target = visual.tracking_target || {{}};
        const bbox = target.bbox || {{}};
        const targetLabel = target.identity || target.label || 'none';
        const targetScore = typeof target.score === 'number' ? target.score.toFixed(2) : '—';
        const targetX = typeof target.target_x === 'number' ? target.target_x.toFixed(2) : '—';
        const missCount = visual.tracking_miss_count ?? 0;
        const trackingDetails = [
          `target ${{targetLabel}}`,
          `score ${{targetScore}}`,
          `x ${{targetX}}`,
          `misses ${{missCount}}`,
          bbox.x_min !== undefined ? `bbox x:${{Number(bbox.x_min || 0).toFixed(2)}}-${{Number(bbox.x_max || 0).toFixed(2)}}` : '',
          visual.tracking_last_outcome_status ? `neck ${{visual.tracking_last_outcome_status}}` : '',
          visual.tracking_age_s !== undefined && visual.tracking_age_s !== null ? `updated ${{fmtSeconds(visual.tracking_age_s)}} ago` : '',
        ].filter(Boolean).join(' · ');
        listItems.push(`<div class="subfunction-item"><div class="sub-top"><strong>Tracking</strong><span class="health-tag ${{healthClass(visual.tracking_last_error ? 'unavailable' : (visual.tracking_status === 'tracking' ? 'healthy' : 'degraded'))}}">${{visual.tracking_status || 'idle'}}</span></div><div class="metric-label">${{visual.tracking_last_error || trackingDetails || 'waiting for tracking state'}}</div></div>`);
      }}
      detections.slice(0, 6).forEach((detection, index) => {{
        const bbox = detection.bbox || {{}};
        listItems.push(`<div class="subfunction-item"><div class="sub-top"><strong>#${{index + 1}} ${{detection.label || 'target'}}</strong><span class="health-tag healthy">${{Number(detection.score || 0).toFixed(2)}}</span></div><div class="metric-label">bbox x:${{Number(bbox.x_min || 0).toFixed(2)}}-${{Number(bbox.x_max || 0).toFixed(2)}} · y:${{Number(bbox.y_min || 0).toFixed(2)}}-${{Number(bbox.y_max || 0).toFixed(2)}}</div></div>`);
      }});
      identityCandidates.slice(0, 4).forEach((candidate) => {{
        const tagHealth = candidate.source === 'session_registration' ? 'healthy' : 'degraded';
        listItems.push(`<div class="subfunction-item"><div class="sub-top"><strong>${{candidate.candidate_id || 'candidate'}}</strong><span class="health-tag ${{tagHealth}}">${{candidate.identity || 'unknown'}}</span></div><div class="metric-label">score ${{Number(candidate.score || 0).toFixed(2)}}</div></div>`);
      }});
      document.getElementById('vision-detection-list').innerHTML = listItems.length
        ? listItems.join('')
        : '<div class="muted">No visual detections yet</div>';
    }}

    async function registerIdentity() {{
      const status = document.getElementById('identity-action-status');
      status.textContent = 'Registering Darrow from current target...';
      try {{
        const response = await fetch('/identity/register', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ display_name: 'Darrow', actor_id: 'darrow' }}),
        }});
        const result = await response.json();
        status.textContent = result.ok ? 'Registered: Darrow' : `Register failed: ${{result.status || result.error || response.status}}`;
        await refresh();
      }} catch (error) {{
        status.textContent = `Register failed: ${{error}}`;
      }}
    }}

    function renderAudio(report) {{
      const audio = report.audio_diagnostics || {{}};
      const dbfs = typeof audio.dbfs === 'number' ? `${{audio.dbfs.toFixed(2)}} dBFS` : '—';
      const rms = typeof audio.rms_level === 'number' ? audio.rms_level.toFixed(3) : '—';
      document.getElementById('audio-summary').innerHTML = [
        ['Capture', audio.capture_health || 'unknown'],
        ['ASR', audio.asr_health || 'unknown'],
        ['Voice', audio.voice_activity ? 'active' : 'idle'],
        ['Level', dbfs],
      ].map(([label, value]) => `<div class="mini-card"><div class="muted">${{label}}</div><div class="metric-value" style="font-size:20px;">${{value}}</div></div>`).join('');

      const items = [];
      items.push(`<div class="subfunction-item"><div class="sub-top"><strong>Input device</strong><span class="health-tag ${{healthClass(audio.capture_health || 'unknown')}}">${{audio.capture_status || audio.capture_health || 'unknown'}}</span></div><div class="metric-label">${{audio.capture_device || 'unknown device'}} · ${{audio.sample_rate || '—'}} Hz · ${{audio.channels || '—'}} ch · chunks=${{audio.chunk_count ?? '—'}} · bytes=${{audio.payload_bytes ?? '—'}}</div></div>`);
      items.push(`<div class="subfunction-item"><div class="sub-top"><strong>Speech window</strong><span class="health-tag ${{healthClass(audio.vad_health || 'unknown')}}">${{audio.vad_status || audio.vad_health || 'unknown'}}</span></div><div class="metric-label">${{audio.speech_window_summary || 'waiting for audio sample'}} · rms=${{rms}}</div></div>`);
      items.push(`<div class="subfunction-item"><div class="sub-top"><strong>Last transcript</strong><span class="health-tag ${{healthClass(audio.asr_health || 'unknown')}}">${{audio.asr_status || audio.asr_health || 'unknown'}}</span></div><div class="metric-label">${{audio.transcript ? audio.transcript : 'No transcript yet'}}</div></div>`);
      document.getElementById('audio-events').innerHTML = items.join('');
    }}

    function renderDialogue(report) {{
      const dialogue = report.dialogue_diagnostics || {{}};
      const latency = dialogue.last_latency_s || {{}};
      const llm = dialogue.last_llm_status || {{}};
      const sessionState = !dialogue.enabled ? 'off' : (!dialogue.running ? 'stopped' : (dialogue.conversation_active ? 'awake' : 'sleeping'));
      document.getElementById('dialogue-summary').innerHTML = [
        ['Loop', dialogue.running ? 'running' : (dialogue.enabled ? 'stopped' : 'off')],
        ['Session', sessionState],
        ['Wake', dialogue.wake_word || '-'],
        ['Sleep', dialogue.sleep_word || '-'],
        ['Phase', dialogue.phase || 'idle'],
        ['Phase age', fmtSeconds(dialogue.current_phase_elapsed_s)],
        ['Status', dialogue.last_status || 'idle'],
        ['Turns', String(dialogue.turn_count ?? 0)],
      ].map(([label, value]) => `<div class="mini-card"><div class="muted">${{label}}</div><div class="metric-value" style="font-size:20px;">${{value}}</div></div>`).join('');

      const transcript = dialogue.last_transcript || 'No transcript yet';
      const reply = dialogue.last_reply || 'No reply yet';
      const error = dialogue.last_error || '';
      const items = [
        `<div class="subfunction-item"><div class="sub-top"><strong>Latency breakdown</strong><span class="health-tag ${{latency.total ? 'healthy' : 'degraded'}}">${{fmtSeconds(latency.total)}}</span></div><div class="metric-label">listen+ASR ${{fmtSeconds(latency.listen_asr)}} · think ${{fmtSeconds(latency.think)}} · speak ${{fmtSeconds(latency.speak)}} · total ${{fmtSeconds(latency.total)}}</div></div>`,
        `<div class="subfunction-item"><div class="sub-top"><strong>Last transcript</strong><span class="health-tag ${{dialogue.last_transcript ? 'healthy' : 'degraded'}}">${{dialogue.phase || 'idle'}}</span></div><div class="metric-label">${{transcript}}</div></div>`,
        `<div class="subfunction-item"><div class="sub-top"><strong>LLM</strong><span class="health-tag ${{healthClass(llm.status === 'ok' ? 'healthy' : (llm.status === 'error' ? 'unavailable' : 'degraded'))}}">${{llm.status || 'idle'}}</span></div><div class="metric-label">${{llm.provider || 'unknown'}} · ${{llm.error || llm.text_preview || 'waiting for first reply'}}</div></div>`,
        `<div class="subfunction-item"><div class="sub-top"><strong>Last reply</strong><span class="health-tag ${{dialogue.last_reply ? 'healthy' : 'degraded'}}">${{dialogue.learning_decision || 'pending'}}</span></div><div class="metric-label">${{reply}}</div></div>`,
      ];
      if (error) {{
        items.push(`<div class="subfunction-item"><div class="sub-top"><strong>Loop error</strong><span class="health-tag unavailable">error</span></div><div class="metric-label">${{error}}</div></div>`);
      }}
      document.getElementById('dialogue-events').innerHTML = items.join('');
    }}

    function renderProbes(report) {{
      const probes = report.probe_metrics || [];
      document.getElementById('probe-table').innerHTML = probes.length
        ? probes.slice(0, 12).map((probe) => `
            <tr>
              <td>
                <strong>${{probe.id}}</strong>
                <div class="muted">${{probe.label || probe.driver || 'probe'}}</div>
              </td>
              <td><span class="health-tag ${{healthClass(probe.health)}}">${{probe.health}}</span></td>
              <td>${{probe.device || probe.model_dir || '—'}}</td>
              <td>${{probe.device ? fmtBool(probe.device_exists) : (probe.missing_file_count ? `missing files:${{probe.missing_file_count}}` : 'ready')}}</td>
              <td>${{fmtLatency(probe.elapsed_ms)}}</td>
            </tr>
          `).join('')
        : '<tr><td colspan="5" class="muted">No probe metrics yet</td></tr>';
    }}

    function renderTimeline(report) {{
      const events = report.recent_traces || [];
      document.getElementById('recent-events').innerHTML = events.length
        ? events.map((event) => `
            <div class="timeline-item">
              <div class="sub-top">
                <strong>${{event.kind || 'unknown'}}</strong>
                <span class="health-tag ${{healthClass(event.status || 'healthy')}}">${{event.status || 'ok'}}</span>
              </div>
              <div class="muted">source=${{event.source || 'unknown'}} · session=${{event.session_id || 'n/a'}} · recorded=${{fmtTime(event.recorded_at_ts)}}</div>
            </div>
          `).join('')
        : '<div class="muted">No recent events captured</div>';
    }}

    function renderChrome(report) {{
      const pill = document.getElementById('system-pill');
      pill.className = `pill ${{report.system_health !== 'healthy' ? 'degraded' : ''}}`;
      document.getElementById('system-pill-text').textContent = `${{report.system_health}} · ${{report.body?.degradation_mode || 'unknown'}}`;
      document.getElementById('refresh-meta').textContent = `Last refresh ${{fmtTime(report.generated_at_ts)}} · warnings ${{(report.warnings || []).length}}`;
      document.getElementById('raw-json').textContent = JSON.stringify(report, null, 2);
    }}

    function render(report) {{
      renderChrome(report);
      renderSummary(report);
      renderOrgans(report);
      renderRuntime(report);
      renderLatencies(report);
      renderWarnings(report);
      renderAudio(report);
      renderDialogue(report);
      renderVision(report);
      renderProbes(report);
      renderTimeline(report);
    }}

    async function refresh() {{
      try {{
        const response = await fetch('/metrics.json', {{ cache: 'no-store' }});
        const report = await response.json();
        render(report);
      }} catch (error) {{
        document.getElementById('refresh-meta').textContent = `Refresh failed: ${{error}}`;
      }}
    }}

    render(INITIAL_REPORT);
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>"""
        rendered = template.replace("{{", "{").replace("}}", "}")
        return rendered.replace("__INITIAL_REPORT__", initial_report)

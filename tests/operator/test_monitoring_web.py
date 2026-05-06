from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import urlopen


def test_monitoring_web_serves_status_and_html() -> None:
    from apps.body_runtime.app import BodyRuntimeApp
    from apps.operator_console.web import MonitoringWebServer

    runtime = BodyRuntimeApp()
    server = MonitoringWebServer(runtime=runtime, host="127.0.0.1", port=0)
    server.start()
    try:
        base_url = f"http://127.0.0.1:{server.port}"
        with urlopen(f"{base_url}/status.json") as response:
            payload = json.loads(response.read().decode("utf-8"))
            cache_control = response.headers.get("Cache-Control")
        with urlopen(f"{base_url}/metrics.json") as response:
            metrics_payload = json.loads(response.read().decode("utf-8"))
        try:
            with urlopen(f"{base_url}/healthz") as response:
                health_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            assert exc.code == 503
            health_payload = json.loads(exc.read().decode("utf-8"))
        with urlopen(base_url) as response:
            html = response.read().decode("utf-8")
    finally:
        server.stop()

    assert payload["body"]["node_id"] == "honjia"
    assert "degradation_mode" in payload["body"]
    assert "warnings" in payload
    assert "organ_cards" in metrics_payload
    assert "latency_metrics" in metrics_payload
    assert "runtime_overview" in metrics_payload
    assert "probe_metrics" in metrics_payload
    assert cache_control == "no-store"
    assert health_payload["system_health"] in {"healthy", "degraded"}
    assert "<title>eibrain honjia monitor</title>" in html
    assert "honjia organ observability" in html
    assert "Hardware probes" in html
    assert "Runtime posture" in html
    assert "Audio diagnostics" in html
    assert "Visual diagnostics" in html
    assert "Vision service" in html
    assert "Neck fusion control" in html
    assert "Backend" in html
    assert "State age" in html
    assert "/metrics.json" in html
    assert "const sessionState = !dialogue.enabled ? 'off'" in html
    assert "['Session', sessionState]" in html
    assert "['Wake', dialogue.wake_word || '-']" in html
    assert "['Sleep', dialogue.sleep_word || '-']" in html
    assert "['Realtime wake', realtimeAudio.running ? 'running'" in html
    assert "Realtime wake audio" in html
    assert "neck_control_diagnostics" in html
    assert "['Desired angle', desiredAngle]" in html
    assert "['Intent count', String(neck.intent_count ?? 0)]" in html


def test_monitoring_web_renders_memory_trace_panel() -> None:
    from apps.operator_console.web import MonitoringWebServer

    class _Runtime:
        def snapshot(self):
            return {"node_id": "honjia", "degradation_mode": "normal", "capabilities": {}, "organs": {}}

        def recent_events(self):
            return []

        def latest_visual_frame_path(self):
            return None

    class _CognitiveRuntime:
        def snapshot(self):
            return {
                "current": {
                    "memory_traces": [
                        {
                            "schema": "eibrain.memory.closed_loop_trace.v1",
                            "round_id": "round-web",
                            "recall": {"count": 1, "items": [{"query": "偏好", "summary": "short"}]},
                            "writeback": {"count": 1, "items": [{"status": "ok", "diagnostics": {"record_id": "mem_web"}}]},
                            "errors": [],
                        }
                    ]
                }
            }

    server = MonitoringWebServer(runtime=_Runtime(), cognitive_runtime=_CognitiveRuntime(), host="127.0.0.1", port=0)
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/metrics.json") as response:
            payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"http://127.0.0.1:{server.port}") as response:
            html = response.read().decode("utf-8")
    finally:
        server.stop()

    assert payload["memory_trace_panel"]["count"] == 1
    assert payload["memory_trace_panel"]["latest"]["round_id"] == "round-web"
    assert "Memory trace" in html
    assert "memory-trace-summary" in html
    assert "memory-trace-events" in html
    assert "function renderMemoryTrace(report)" in html


def test_monitoring_web_serves_latest_vision_frame(tmp_path) -> None:
    from apps.operator_console.web import MonitoringWebServer

    frame_path = tmp_path / "latest.jpg"
    frame_path.write_bytes(b"jpeg-bytes")

    class _Runtime:
        def snapshot(self):
            return {"node_id": "honjia", "degradation_mode": "normal", "capabilities": {}, "organs": {}}

        def recent_events(self):
            return []

        def latest_visual_frame_path(self):
            return str(frame_path)

    server = MonitoringWebServer(runtime=_Runtime(), host="127.0.0.1", port=0)
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/vision/latest.jpg?ts=1") as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type")
    finally:
        server.stop()

    assert payload == b"jpeg-bytes"
    assert content_type == "image/jpeg"



def test_monitoring_web_healthz_stays_200_when_report_is_degraded() -> None:
    from apps.operator_console.web import MonitoringWebServer

    class _Runtime:
        def snapshot(self):
            return {
                "node_id": "honjia",
                "degradation_mode": "low_confidence_body",
                "capabilities": {"can_transcribe_speech": False},
                "organs": {"ear": {"health": "degraded", "subfunctions": {}}},
            }

        def recent_events(self):
            return []

        def latest_visual_frame_path(self):
            return None

    server = MonitoringWebServer(runtime=_Runtime(), host="127.0.0.1", port=0)
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/healthz") as response:
            payload = json.loads(response.read().decode("utf-8"))
            status_code = response.status
    finally:
        server.stop()

    assert status_code == 200
    assert payload["system_health"] == "degraded"

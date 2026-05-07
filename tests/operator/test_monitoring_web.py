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


def test_monitoring_web_surfaces_audio_live_trace_readiness_without_secret_leaks(monkeypatch) -> None:
    from apps.operator_console.web import MonitoringWebServer

    fake_key = "dashscope-secret-for-monitor"
    monkeypatch.setenv("MINIMAX_API_KEY", fake_key)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("EIVOICE_DASHSCOPE_API_KEY", raising=False)

    class _Runtime:
        def snapshot(self):
            return {
                "node_id": "honjia",
                "degradation_mode": "normal",
                "capabilities": {},
                "organs": {},
                "audio_frontend": {
                    "aec": {"enabled": True, "available": False},
                    "loopback": {"enabled": True, "available": True},
                    "lastCapture": {
                        "loopbackReference": {
                            "ready": False,
                            "state": "aec_unavailable",
                            "reason": "aec_unavailable",
                        }
                    },
                },
                "voice_dialogue": {
                    "enabled": True,
                    "running": True,
                    "voice_chain_benchmark": {
                        "turnCount": 2,
                        "roundLeakCount": 1,
                        "metrics": {
                            "interruptStopMs": {"p95": 410.0, "threshold": 300.0, "pass": False},
                        },
                        "bottleneck": {"field": "interruptStopMs", "p95": 410.0, "threshold": 300.0},
                    },
                },
            }

        def recent_events(self):
            return []

        def latest_visual_frame_path(self):
            return None

    server = MonitoringWebServer(runtime=_Runtime(), host="127.0.0.1", port=0)
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/status.json") as response:
            payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"http://127.0.0.1:{server.port}/livez") as response:
            live_payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"http://127.0.0.1:{server.port}") as response:
            html = response.read().decode("utf-8")
    finally:
        server.stop()

    audio = payload["audio_diagnostics"]
    assert audio["live_trace_summary"] == "not ready: interruptStopMs"
    assert audio["provider_readiness"] == "degraded"
    assert audio["aec_readiness"] == "unavailable"
    assert audio["interrupt_stop_ready"] is False
    assert audio["round_leak_count"] == 1
    assert fake_key not in json.dumps(payload)
    assert fake_key not in html
    assert "Live trace" in html
    assert "Provider readiness" in html
    assert "AEC readiness" in html
    assert "Interrupt stop" in html
    assert "Round leak" in html
    assert "audio.live_trace_summary || 'waiting'" in html
    assert "audio.provider_readiness || 'unknown'" in html
    assert "audio.aec_readiness || 'unknown'" in html


def test_monitoring_web_surfaces_visual_long_tracking_and_memory_guardrails() -> None:
    from apps.operator_console.web import MonitoringWebServer

    class _Runtime:
        def snapshot(self):
            return {
                "node_id": "honjia",
                "degradation_mode": "normal",
                "capabilities": {"can_see_people": True},
                "visual_tracking": {
                    "running": True,
                    "status": "tracking",
                    "tracking_stability": {"state": "stable", "score": 0.9},
                    "switch_count": 1,
                    "lost_count": 0,
                    "reacquired_count": 1,
                    "scene_graph": {"summary": "person beside desk"},
                    "pose": {"available": True, "summary": "seated"},
                    "clipLabels": [{"label": "person beside desk", "score": 0.84}],
                    "depth": {"status": "waiting"},
                    "soak_summary": {"stable_ratio": 0.96, "switch_count": 1},
                },
                "organs": {
                    "eye": {
                        "health": "healthy",
                        "subfunctions": {
                            "camera": {"health": "healthy", "details": {"frame_path": "/tmp/latest.jpg"}},
                            "detection": {"health": "healthy", "details": {"status": "live", "detections": []}},
                        },
                    }
                },
            }

        def recent_events(self):
            return []

        def latest_visual_frame_path(self):
            return None

    class _CognitiveRuntime:
        def snapshot(self):
            return {
                "memory_diagnostics": {
                    "conflict_summary": "identity conflict blocked",
                    "persona_guardrail": {"status": "blocked", "summary": "persona guardrail active"},
                }
            }

    server = MonitoringWebServer(runtime=_Runtime(), cognitive_runtime=_CognitiveRuntime(), host="127.0.0.1", port=0)
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/status.json") as response:
            payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"http://127.0.0.1:{server.port}/livez") as response:
            live_payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"http://127.0.0.1:{server.port}") as response:
            html = response.read().decode("utf-8")
    finally:
        server.stop()

    visual = payload["visual_diagnostics"]
    memory = payload["memory_diagnostics"]
    assert visual["tracking_stability"]["state"] == "stable"
    assert visual["scene_graph_summary"] == "person beside desk"
    assert visual["pose_availability"] == "present"
    assert visual["clip_availability"] == "present"
    assert memory["memory_conflict_summary"] == "identity conflict blocked"
    assert memory["persona_guardrail_status"] == "blocked"
    assert payload["system_health"] == "degraded"
    assert "persona.guardrail=blocked" in payload["degraded_reasons"]
    assert live_payload["system_health"] == "degraded"
    assert live_payload["degraded_reasons"] == payload["degraded_reasons"]
    assert "Tracking stability" in html
    assert "Switch/Lost/Reacquired" in html
    assert "Scene graph" in html
    assert "Pose/CLIP/Semantic/Depth/Distance/Tracking" in html
    assert "Memory conflict" in html
    assert "Persona guardrail" in html
    assert "const multimodalHealth" in html
    assert "multimodalPresentCount > 0 ? 'degraded' : 'waiting'" in html
    assert "visual.scene_graph_summary || 'waiting'" in html
    assert "memory.memory_conflict_summary || 'unknown'" in html


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



def test_monitoring_web_healthz_returns_503_when_report_is_degraded_while_status_stays_200() -> None:
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
        with urlopen(f"http://127.0.0.1:{server.port}/status.json") as response:
            status_payload = json.loads(response.read().decode("utf-8"))
            status_code = response.status
        with urlopen(f"http://127.0.0.1:{server.port}/metrics.json") as response:
            metrics_payload = json.loads(response.read().decode("utf-8"))
            metrics_code = response.status
        with urlopen(f"http://127.0.0.1:{server.port}/livez") as response:
            live_payload = json.loads(response.read().decode("utf-8"))
            live_code = response.status
        try:
            with urlopen(f"http://127.0.0.1:{server.port}/healthz") as response:
                health_payload = json.loads(response.read().decode("utf-8"))
                health_code = response.status
        except HTTPError as exc:
            health_payload = json.loads(exc.read().decode("utf-8"))
            health_code = exc.code
    finally:
        server.stop()

    assert status_code == 200
    assert metrics_code == 200
    assert live_code == 200
    assert health_code == 503
    assert status_payload["system_health"] == "degraded"
    assert metrics_payload["system_health"] == "degraded"
    assert live_payload["status"] == "alive"
    assert live_payload["system_health"] == "degraded"
    assert "degradation_mode=low_confidence_body" in live_payload["degraded_reasons"]
    assert health_payload["system_health"] == "degraded"

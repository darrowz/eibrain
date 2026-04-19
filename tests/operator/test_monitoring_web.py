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
    assert "/metrics.json" in html

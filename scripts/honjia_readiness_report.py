#!/usr/bin/env python3
"""Offline-capable honjia body-runtime readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.body_runtime.voice_chain_selftest import run_voice_chain_selftest
from apps.body_runtime.voice_provider_smoke import build_report as build_voice_provider_report
from apps.body_runtime.vision_soak import DEFAULT_STATUS_URL
from apps.body_runtime.vision_soak import normalize_vision_status_sample
from apps.body_runtime.vision_soak import run_synthetic_vision_soak
from apps.body_runtime.vision_soak import run_vision_soak
from apps.body_runtime.vision_soak import summarize_vision_soak


SCHEMA = "eibrain.honjia_readiness_report.v1"
DEFAULT_SYSTEMD_SERVICES = (
    "eihead-monitor.service",
    "eihead-runtime.service",
    "eibrain-vision-hailo.service",
    "eibrain-monitor.service",
)


def build_report(
    *,
    live: bool = False,
    status_url: str = DEFAULT_STATUS_URL,
    status_file: Path | None = None,
    duration_s: float = 15.0,
    interval_s: float = 5.0,
    require_provider_config: bool = False,
    services: Sequence[str] | None = None,
) -> dict[str, Any]:
    voice_chain = run_voice_chain_selftest()
    provider_smoke = build_voice_provider_report("all", dry_run=True, live=False)
    vision = _vision_report(
        live=live,
        status_url=status_url,
        status_file=status_file,
        duration_s=duration_s,
        interval_s=interval_s,
    )
    service_names = tuple(services or (DEFAULT_SYSTEMD_SERVICES if live else ()))
    service_reports = [_systemd_service_report(name) for name in service_names]
    checks = _checks(
        live=live,
        voice_chain=voice_chain,
        provider_smoke=provider_smoke,
        vision=vision,
        service_reports=service_reports,
    )
    required_failures = [
        name
        for name in ("voice_chain_selftest", "vision_soak")
        if checks.get(name, {}).get("ok") is not True
    ]
    if live and checks.get("monitor_active", {}).get("ok") is not True:
        required_failures.append("monitor_active")
    if require_provider_config and checks.get("voice_provider_configured", {}).get("ok") is not True:
        required_failures.append("voice_provider_configured")
    optional_issues = [
        name
        for name, check in checks.items()
        if check.get("ok") is False and name not in required_failures
    ]
    overall = {
        "pass": not required_failures,
        "status": "ok" if not required_failures else "failed",
        "required_failures": required_failures,
        "optional_issues": optional_issues,
    }
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "live": bool(live),
        "offline_capable": True,
        "inputs": {
            "status_url": status_url if live and not status_file else None,
            "status_file": str(status_file) if status_file is not None else None,
            "duration_s": float(duration_s),
            "interval_s": float(interval_s),
            "require_provider_config": bool(require_provider_config),
            "services": list(service_names),
        },
        "voice_chain_selftest": voice_chain,
        "voice_provider_smoke": provider_smoke,
        "vision_soak": vision,
        "services": service_reports,
        "checks": checks,
        "overall": overall,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline-capable honjia deployment readiness report")
    parser.add_argument("--live", action="store_true", help="Read local monitor/systemd evidence in addition to offline checks.")
    parser.add_argument("--status-url", default=DEFAULT_STATUS_URL, help="Monitor status URL for --live mode.")
    parser.add_argument("--status-file", type=Path, default=None, help="Read status JSON from a local file instead of HTTP.")
    parser.add_argument("--duration", type=float, default=15.0, help="Vision soak duration for live status sampling.")
    parser.add_argument("--interval", type=float, default=5.0, help="Vision soak sample interval for live status sampling.")
    parser.add_argument("--require-provider-config", action="store_true", help="Fail the report when cloud provider env is not configured.")
    parser.add_argument("--service", action="append", default=[], help="Additional systemd services to inspect with systemctl show.")
    parser.add_argument("--pretty", action="store_true", help="Print indented JSON.")
    args = parser.parse_args(argv)

    report = build_report(
        live=bool(args.live),
        status_url=str(args.status_url),
        status_file=args.status_file,
        duration_s=float(args.duration),
        interval_s=float(args.interval),
        require_provider_config=bool(args.require_provider_config),
        services=tuple(args.service),
    )
    indent = 2 if args.pretty else None
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0 if report["overall"]["pass"] else 1


def _vision_report(
    *,
    live: bool,
    status_url: str,
    status_file: Path | None,
    duration_s: float,
    interval_s: float,
) -> dict[str, Any]:
    if status_file is not None:
        payload = _load_json_object(status_file)
        summary = summarize_vision_soak([normalize_vision_status_sample(payload)])
        summary["collection"] = {
            "source": "status_file",
            "status_file": str(status_file),
            "requested_duration_s": 0.0,
            "duration_s": 0.0,
            "interval_s": 0.0,
            "error_count": 0,
            "errors": [],
        }
        return summary
    if live:
        try:
            summary = run_vision_soak(
                duration_s=max(0.0, float(duration_s)),
                interval_s=max(0.1, float(interval_s)),
                status_url=status_url,
            )
        except Exception as exc:  # noqa: BLE001 - readiness must degrade into JSON instead of crashing.
            return {
                "pass": False,
                "sample_count": 0,
                "fail_reason": "status_unavailable",
                "bottleneck_reason": "status_unavailable",
                "readiness": {
                    "hailo_fps": {"ok": False, "observed_ratio": 0.0, "threshold_ratio": 0.0},
                    "hailo_drop_rate": {"ok": False, "observed": 0.0, "threshold": 0.0},
                    "hailo_frame_age": {"ok": False, "observed_p95_ms": 0.0, "threshold_ms": 0.0},
                    "monitor_active": {"ok": False, "status": "error", "observed_ratio": 0.0},
                },
                "collection": {
                    "source": "status_url",
                    "status_url": status_url,
                    "requested_duration_s": float(duration_s),
                    "duration_s": 0.0,
                    "interval_s": float(interval_s),
                    "error_count": 1,
                    "errors": [{"type": type(exc).__name__, "message": str(exc)}],
                },
            }
        summary.setdefault("collection", {})
        summary["collection"]["source"] = "status_url"
        summary["collection"]["status_url"] = status_url
        return summary
    summary = run_synthetic_vision_soak()
    summary["collection"] = {
        "source": "synthetic",
        "requested_duration_s": 0.0,
        "duration_s": 0.0,
        "interval_s": 0.0,
        "error_count": 0,
        "errors": [],
    }
    return summary


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"status payload must be a JSON object: {path}")
    return dict(payload)


def _systemd_service_report(service: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "show",
                service,
                "--no-pager",
                "--property=Id,LoadState,ActiveState,SubState,NRestarts,ExecMainStatus,ExecMainPID,ActiveEnterTimestamp,ExecMainStartTimestamp",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {"service": service, "checked": False, "status": "systemctl_unavailable"}
    if completed.returncode != 0:
        return {
            "service": service,
            "checked": False,
            "status": "systemctl_error",
            "returncode": completed.returncode,
            "stderr": (completed.stderr or "").strip(),
        }

    properties: dict[str, str] = {}
    for line in (completed.stdout or "").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    restart_count = _safe_int(properties.get("NRestarts"))
    active_state = properties.get("ActiveState", "")
    sub_state = properties.get("SubState", "")
    return {
        "service": service,
        "checked": True,
        "status": "ok",
        "load_state": properties.get("LoadState", ""),
        "active_state": active_state,
        "sub_state": sub_state,
        "restart_count": restart_count,
        "restart_observed": restart_count > 0,
        "main_pid": _safe_int(properties.get("ExecMainPID")),
        "exec_main_status": _safe_int(properties.get("ExecMainStatus")),
        "active_enter_timestamp": properties.get("ActiveEnterTimestamp", ""),
        "exec_main_start_timestamp": properties.get("ExecMainStartTimestamp", ""),
    }


def _checks(
    *,
    live: bool,
    voice_chain: Mapping[str, Any],
    provider_smoke: Mapping[str, Any],
    vision: Mapping[str, Any],
    service_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    monitor = vision.get("readiness", {})
    monitor_active = monitor.get("monitor_active", {}) if isinstance(monitor, Mapping) else {}
    checked_service_reports = [item for item in service_reports if item.get("checked") is True]
    restart_observed = any(item.get("restart_observed") is True for item in checked_service_reports)
    return {
        "voice_chain_selftest": {
            "ok": bool(voice_chain.get("codeReady")),
            "failed_metrics": list(voice_chain.get("failedMetrics", [])),
        },
        "voice_provider_configured": {
            "ok": bool(provider_smoke.get("configured")),
            "configured_provider_count": (
                provider_smoke.get("readiness", {}).get("configured_provider_count", 0)
                if isinstance(provider_smoke.get("readiness"), Mapping)
                else 0
            ),
        },
        "vision_soak": {
            "ok": bool(vision.get("pass")),
            "fail_reason": vision.get("fail_reason") or vision.get("bottleneck_reason"),
        },
        "monitor_active": {
            "ok": monitor_active.get("ok") if isinstance(monitor_active, Mapping) else (False if live else None),
            "status": monitor_active.get("status") if isinstance(monitor_active, Mapping) else ("not_checked" if not live else "unknown"),
        },
        "service_restart_evidence": {
            "ok": True if checked_service_reports else None,
            "checked_service_count": len(checked_service_reports),
            "restart_observed": restart_observed,
        },
    }


def _safe_int(value: Any) -> int:
    try:
        return int(str(value).strip() or "0")
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

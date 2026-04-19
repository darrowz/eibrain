"""Operator console placeholder."""

from __future__ import annotations

import time
from typing import Any


class OperatorConsoleApp:
    """Operator console for status summaries."""

    IMPORTANT_CAPABILITIES = (
        "can_hear_voice",
        "can_transcribe_speech",
        "can_see_people",
        "can_speak",
        "can_orient_head",
    )

    ORGAN_LABELS = {
        "ear": "Ear",
        "eye": "Eye",
        "mouth": "Mouth",
        "neck": "Neck",
    }

    def build_status_report(
        self,
        *,
        body_snapshot: dict[str, object],
        cognitive_snapshot: dict[str, object],
        traces: list[dict[str, object]],
    ) -> dict[str, object]:
        generated_at = time.time()
        degradation = str(body_snapshot.get("degradation_mode", "unknown"))
        capabilities = dict(body_snapshot.get("capabilities", {}))
        organs = dict(body_snapshot.get("organs", {}))
        warnings = [
            f"{name}=false"
            for name in self.IMPORTANT_CAPABILITIES
            if capabilities.get(name) is False
        ]
        degraded_organs = sorted(
            organ_name
            for organ_name, snapshot in organs.items()
            if isinstance(snapshot, dict) and snapshot.get("health") != "healthy"
        )
        organ_cards = self._build_organ_cards(organs)
        latency_metrics = self._build_latency_metrics(organs)
        capability_status = self._build_capability_status(capabilities)
        probe_metrics = self._build_probe_metrics(organs)
        runtime_overview = self._build_runtime_overview(
            body_snapshot=body_snapshot,
            organ_cards=organ_cards,
            probe_metrics=probe_metrics,
        )
        driver_breakdown = self._build_driver_breakdown(probe_metrics)
        visual_diagnostics = self._build_visual_diagnostics(organs)
        summary = self._build_summary(
            capabilities=capabilities,
            warnings=warnings,
            degraded_organs=degraded_organs,
            latency_metrics=latency_metrics,
            runtime_overview=runtime_overview,
            probe_metrics=probe_metrics,
        )
        system_health = "healthy" if degradation == "normal" and not warnings and not degraded_organs else "degraded"
        return {
            "system_health": system_health,
            "generated_at_ts": generated_at,
            "trace_count": len(traces),
            "warnings": warnings,
            "degraded_organs": degraded_organs,
            "summary": summary,
            "runtime_overview": runtime_overview,
            "capability_status": capability_status,
            "driver_breakdown": driver_breakdown,
            "probe_metrics": probe_metrics,
            "visual_diagnostics": visual_diagnostics,
            "organ_cards": organ_cards,
            "latency_metrics": latency_metrics,
            "event_breakdown": self._build_event_breakdown(traces),
            "body": body_snapshot,
            "cognition": cognitive_snapshot,
            "recent_traces": traces[-5:],
        }

    def _build_organ_cards(self, organs: dict[str, object]) -> list[dict[str, object]]:
        cards: list[dict[str, object]] = []
        for organ_name, snapshot in organs.items():
            if not isinstance(snapshot, dict):
                continue
            subfunctions = snapshot.get("subfunctions", {})
            if not isinstance(subfunctions, dict):
                subfunctions = {}
            entries: list[dict[str, object]] = []
            latencies: list[float] = []
            for sub_name, sub_snapshot in subfunctions.items():
                if not isinstance(sub_snapshot, dict):
                    continue
                details = sub_snapshot.get("details", {})
                if not isinstance(details, dict):
                    details = {}
                probe_details = self._extract_probe_details(details)
                elapsed_ms = details.get("elapsed_ms")
                if isinstance(elapsed_ms, (int, float)):
                    latencies.append(float(elapsed_ms))
                visual_summary = details.get("scene_summary") or details.get("identity_summary")
                entries.append(
                    {
                        "name": sub_name,
                        "health": sub_snapshot.get("health", "unknown"),
                        "driver": details.get("driver", "unknown"),
                        "elapsed_ms": elapsed_ms,
                        "status": details.get("status", sub_snapshot.get("health", "unknown")),
                        "error": details.get("error") or details.get("reason") or details.get("stderr", ""),
                        "visual_summary": visual_summary,
                        "probe": probe_details,
                    }
                )
            cards.append(
                {
                    "name": organ_name,
                    "label": self.ORGAN_LABELS.get(organ_name, organ_name.title()),
                    "health": snapshot.get("health", "unknown"),
                    "subfunction_count": len(entries),
                    "healthy_subfunctions": sum(1 for entry in entries if entry["health"] == "healthy"),
                    "degraded_subfunctions": sum(1 for entry in entries if entry["health"] != "healthy"),
                    "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
                    "max_latency_ms": round(max(latencies), 2) if latencies else None,
                    "subfunctions": entries,
                }
            )
        return cards

    def _build_latency_metrics(self, organs: dict[str, object]) -> list[dict[str, object]]:
        metrics: list[dict[str, object]] = []
        for organ_name, snapshot in organs.items():
            if not isinstance(snapshot, dict):
                continue
            subfunctions = snapshot.get("subfunctions", {})
            if not isinstance(subfunctions, dict):
                continue
            for sub_name, sub_snapshot in subfunctions.items():
                if not isinstance(sub_snapshot, dict):
                    continue
                details = sub_snapshot.get("details", {})
                if not isinstance(details, dict):
                    continue
                elapsed_ms = details.get("elapsed_ms")
                if not isinstance(elapsed_ms, (int, float)):
                    continue
                metrics.append(
                    {
                        "id": f"{organ_name}.{sub_name}",
                        "organ": organ_name,
                        "subfunction": sub_name,
                        "driver": details.get("driver", "unknown"),
                        "health": sub_snapshot.get("health", "unknown"),
                        "elapsed_ms": round(float(elapsed_ms), 2),
                    }
                )
        metrics.sort(key=lambda item: float(item["elapsed_ms"]), reverse=True)
        return metrics

    def _build_summary(
        self,
        *,
        capabilities: dict[str, object],
        warnings: list[str],
        degraded_organs: list[str],
        latency_metrics: list[dict[str, object]],
        runtime_overview: dict[str, object],
        probe_metrics: list[dict[str, object]],
    ) -> dict[str, object]:
        enabled_capabilities = sum(1 for value in capabilities.values() if value is True)
        avg_latency = round(
            sum(float(metric["elapsed_ms"]) for metric in latency_metrics) / len(latency_metrics),
            2,
        ) if latency_metrics else None
        p95_latency = None
        if latency_metrics:
            ordered = sorted(float(metric["elapsed_ms"]) for metric in latency_metrics)
            index = min(len(ordered) - 1, max(0, int(len(ordered) * 0.95) - 1))
            p95_latency = round(ordered[index], 2)
        return {
            "enabled_capability_count": enabled_capabilities,
            "capability_count": len(capabilities),
            "warning_count": len(warnings),
            "degraded_organ_count": len(degraded_organs),
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "healthy_subfunction_count": runtime_overview["healthy_subfunction_count"],
            "subfunction_count": runtime_overview["subfunction_count"],
            "real_driver_count": runtime_overview["real_driver_count"],
            "noop_driver_count": runtime_overview["noop_driver_count"],
            "unavailable_probe_count": sum(1 for probe in probe_metrics if probe["health"] == "unavailable"),
        }

    def _build_runtime_overview(
        self,
        *,
        body_snapshot: dict[str, object],
        organ_cards: list[dict[str, object]],
        probe_metrics: list[dict[str, object]],
    ) -> dict[str, object]:
        subfunction_count = sum(int(card["subfunction_count"]) for card in organ_cards)
        healthy_subfunction_count = sum(int(card["healthy_subfunctions"]) for card in organ_cards)
        return {
            "node_id": body_snapshot.get("node_id", "unknown"),
            "degradation_mode": body_snapshot.get("degradation_mode", "unknown"),
            "organ_count": body_snapshot.get("organ_count", len(organ_cards)),
            "recent_event_count": body_snapshot.get("recent_event_count", 0),
            "subfunction_count": subfunction_count,
            "healthy_subfunction_count": healthy_subfunction_count,
            "real_driver_count": sum(1 for probe in probe_metrics if probe["driver"] != "noop"),
            "noop_driver_count": sum(1 for probe in probe_metrics if probe["driver"] == "noop"),
        }

    def _build_capability_status(self, capabilities: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "name": name,
                "enabled": bool(value),
                "status": "enabled" if value else "disabled",
            }
            for name, value in sorted(capabilities.items())
        ]

    def _build_driver_breakdown(self, probe_metrics: list[dict[str, object]]) -> list[dict[str, object]]:
        counts: dict[str, int] = {}
        for probe in probe_metrics:
            driver = str(probe.get("driver", "unknown"))
            counts[driver] = counts.get(driver, 0) + 1
        return [
            {"driver": driver, "count": count}
            for driver, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

    def _build_probe_metrics(self, organs: dict[str, object]) -> list[dict[str, object]]:
        probes: list[dict[str, object]] = []
        for organ_name, snapshot in organs.items():
            if not isinstance(snapshot, dict):
                continue
            subfunctions = snapshot.get("subfunctions", {})
            if not isinstance(subfunctions, dict):
                continue
            for sub_name, sub_snapshot in subfunctions.items():
                if not isinstance(sub_snapshot, dict):
                    continue
                details = sub_snapshot.get("details", {})
                if not isinstance(details, dict):
                    details = {}
                probe = self._extract_probe_details(details)
                probe.update(
                    {
                        "id": f"{organ_name}.{sub_name}",
                        "organ": organ_name,
                        "subfunction": sub_name,
                        "driver": details.get("driver", "unknown"),
                        "health": sub_snapshot.get("health", "unknown"),
                        "status": details.get("status", sub_snapshot.get("health", "unknown")),
                        "elapsed_ms": round(float(details["elapsed_ms"]), 2)
                        if isinstance(details.get("elapsed_ms"), (int, float))
                        else None,
                    }
                )
                probes.append(probe)
        probes.sort(key=self._probe_sort_key)
        return probes

    def _build_visual_diagnostics(self, organs: dict[str, object]) -> dict[str, object]:
        eye = organs.get("eye", {})
        if not isinstance(eye, dict):
            return {"enabled": False, "detections": [], "identity_candidates": []}
        subfunctions = eye.get("subfunctions", {})
        if not isinstance(subfunctions, dict):
            subfunctions = {}
        camera = subfunctions.get("camera", {})
        detection = subfunctions.get("detection", {})
        identity = subfunctions.get("identity", {})
        if not isinstance(camera, dict):
            camera = {}
        if not isinstance(detection, dict):
            detection = {}
        if not isinstance(identity, dict):
            identity = {}
        camera_details = dict(camera.get("details", {})) if isinstance(camera.get("details", {}), dict) else {}
        detection_details = dict(detection.get("details", {})) if isinstance(detection.get("details", {}), dict) else {}
        identity_details = dict(identity.get("details", {})) if isinstance(identity.get("details", {}), dict) else {}
        detections = detection_details.get("detections", [])
        identity_candidates = identity_details.get("identity_candidates", [])
        if not isinstance(detections, list):
            detections = []
        if not isinstance(identity_candidates, list):
            identity_candidates = []
        frame_path = detection_details.get("frame_path") or camera_details.get("frame_path")
        frame_captured_at_ts = detection_details.get("frame_captured_at_ts") or camera_details.get("frame_captured_at_ts")
        return {
            "enabled": bool(frame_path or detections or identity_candidates),
            "frame_available": bool(frame_path),
            "frame_url": "/vision/latest.jpg" if frame_path else None,
            "frame_captured_at_ts": frame_captured_at_ts,
            "camera_health": camera.get("health", "unknown"),
            "detection_health": detection.get("health", "unknown"),
            "identity_health": identity.get("health", "unknown"),
            "detection_status": detection_details.get("status", detection.get("health", "unknown")),
            "identity_status": identity_details.get("status", identity.get("health", "unknown")),
            "detection_count": len(detections),
            "detections": detections,
            "identity_candidates": identity_candidates,
            "scene_summary": detection_details.get("scene_summary", "no visual diagnostics yet"),
            "identity_summary": identity_details.get("identity_summary", "identity chain inactive"),
            "scene_labels": detection_details.get("scene_labels", []),
            "top_detection": detection_details.get("top_detection"),
        }

    @staticmethod
    def _probe_sort_key(probe: dict[str, object]) -> tuple[int, str]:
        priority = {
            "unavailable": 0,
            "degraded": 1,
            "healthy": 2,
        }.get(str(probe.get("health", "unknown")), 3)
        return (priority, str(probe.get("id", "")))

    @staticmethod
    def _extract_probe_details(details: dict[str, Any]) -> dict[str, object]:
        nested_details = details.get("details", {})
        if not isinstance(nested_details, dict):
            nested_details = {}
        missing_files = nested_details.get("missing_files")
        if not isinstance(missing_files, list):
            missing_files = []
        return {
            "label": nested_details.get("label") or nested_details.get("driver") or details.get("driver"),
            "device": nested_details.get("device"),
            "device_exists": nested_details.get("device_exists"),
            "binary": nested_details.get("binary"),
            "model_dir": nested_details.get("model_dir"),
            "missing_files": missing_files,
            "missing_file_count": len(missing_files),
        }

    @staticmethod
    def _build_event_breakdown(traces: list[dict[str, object]]) -> list[dict[str, object]]:
        counts: dict[str, int] = {}
        for trace in traces:
            kind = str(trace.get("kind", "unknown"))
            counts[kind] = counts.get(kind, 0) + 1
        return [
            {"kind": kind, "count": count}
            for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ]

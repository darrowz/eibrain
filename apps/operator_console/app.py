"""Operator console status aggregation."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from eibrain.voice.readiness import build_voice_chain_readiness


class OperatorConsoleApp:
    """Operator console for status summaries."""

    PAN_MOTION_PROOF_PATH = Path("/tmp/eibrain-pan-proof/summary.json")

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
        trace_details = self._latest_audio_trace_details(traces)
        self._annotate_live_voice_loop(
            organs,
            body_snapshot=body_snapshot,
            trace_details=trace_details,
        )
        organ_cards = self._build_organ_cards(organs)
        latency_metrics = self._build_latency_metrics(organs)
        if not latency_metrics:
            latency_metrics = self._build_trace_latency_metrics(traces)
        latency_metrics.extend(
            self._build_dialogue_latency_metrics(
                body_snapshot=body_snapshot,
                cognitive_snapshot=cognitive_snapshot,
            )
        )
        latency_metrics.sort(key=lambda item: float(item["elapsed_ms"]), reverse=True)
        capability_status = self._build_capability_status(capabilities)
        probe_metrics = self._build_probe_metrics(organs)
        runtime_overview = self._build_runtime_overview(
            body_snapshot=body_snapshot,
            organ_cards=organ_cards,
            probe_metrics=probe_metrics,
        )
        driver_breakdown = self._build_driver_breakdown(probe_metrics)
        audio_diagnostics = self._build_audio_diagnostics(organs, traces=traces)
        visual_diagnostics = self._build_visual_diagnostics(
            body_snapshot=body_snapshot,
            organs=organs,
        )
        dialogue_diagnostics = self._build_dialogue_diagnostics(
            body_snapshot=body_snapshot,
            cognitive_snapshot=cognitive_snapshot,
        )
        neck_control_diagnostics = self._build_neck_control_diagnostics(body_snapshot=body_snapshot)
        memory_snapshot = self._build_memory_monitor_snapshot(
            body_snapshot=body_snapshot,
            cognitive_snapshot=cognitive_snapshot,
        )
        memory_trace_panel = self._build_memory_trace_panel(memory_snapshot)
        memory_diagnostics = self._build_memory_diagnostics(
            memory_snapshot,
            memory_trace_panel=memory_trace_panel,
        )
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
            "audio_diagnostics": audio_diagnostics,
            "visual_diagnostics": visual_diagnostics,
            "dialogue_diagnostics": dialogue_diagnostics,
            "neck_control_diagnostics": neck_control_diagnostics,
            "memory_diagnostics": memory_diagnostics,
            "memory_trace_panel": memory_trace_panel,
            "organ_cards": organ_cards,
            "latency_metrics": latency_metrics,
            "event_breakdown": self._build_event_breakdown(traces),
            "body": body_snapshot,
            "cognition": cognitive_snapshot,
            "recent_traces": traces[-5:],
        }

    @staticmethod
    def _annotate_live_voice_loop(
        organs: dict[str, object],
        *,
        body_snapshot: dict[str, object],
        trace_details: dict[str, object] | None = None,
    ) -> None:
        loop = body_snapshot.get("voice_dialogue", {})
        if not isinstance(loop, dict) or not loop.get("running"):
            return
        ear = organs.get("ear")
        if not isinstance(ear, dict):
            return
        subfunctions = ear.get("subfunctions", {})
        if not isinstance(subfunctions, dict):
            return
        for sub_name, sub_snapshot in subfunctions.items():
            if not isinstance(sub_snapshot, dict):
                continue
            details = sub_snapshot.get("details", {})
            if not isinstance(details, dict):
                details = {}
                sub_snapshot["details"] = details
            if details.get("status") == "live_probe_skipped":
                details["status"] = "listening_loop"
            details["driver"] = "voice_dialogue_loop"
            details.setdefault("listening", True)
            details.setdefault("voice_loop_phase", loop.get("phase"))
            details.setdefault("voice_loop_status", loop.get("last_status"))
            details.setdefault("updated_at_ts", loop.get("updated_at_ts"))

            if not isinstance(trace_details, dict):
                continue

            if sub_name == "capture":
                capture_elapsed_ms = OperatorConsoleApp._first_numeric_value(
                    trace_details,
                    keys=("capture_elapsed_ms", "capture_window_elapsed_ms", "capture_read_elapsed_ms", "capture_latency_ms"),
                )
                if isinstance(capture_elapsed_ms, (int, float)):
                    details["elapsed_ms"] = round(float(capture_elapsed_ms), 2)
                details.setdefault("status", "recent_trace")
                for key in (
                    "sample_rate",
                    "channels",
                    "chunk_count",
                    "payload_bytes",
                    "dbfs",
                    "rms_level",
                    "peak_level",
                    "voice_activity",
                    "streaming_vad",
                    "vad_triggered",
                    "speech_window_summary",
                    "capture_device",
                    "recorded_at_ts",
                ):
                    value = trace_details.get(key)
                    if value is not None and details.get(key) is None:
                        details[key] = value

            if sub_name == "asr":
                transcript_from_trace = str(trace_details.get("text", "") or "")
                if transcript_from_trace:
                    details["transcript"] = transcript_from_trace
                asr_status = trace_details.get("asr_status")
                if asr_status:
                    details["status"] = asr_status
                asr_decode_elapsed_ms = OperatorConsoleApp._first_numeric_value(
                    trace_details,
                    keys=("asr_decode_elapsed_ms", "asr_infer_elapsed_ms"),
                )
                asr_elapsed_ms = OperatorConsoleApp._first_numeric_value(
                    trace_details,
                    keys=("asr_elapsed_ms", "asr_decode_elapsed_ms", "asr_infer_elapsed_ms"),
                )
                if isinstance(asr_decode_elapsed_ms, (int, float)):
                    details["asr_decode_elapsed_ms"] = round(float(asr_decode_elapsed_ms), 2)
                if isinstance(asr_elapsed_ms, (int, float)):
                    if isinstance(asr_decode_elapsed_ms, (int, float)):
                        details["elapsed_ms"] = round(float(asr_decode_elapsed_ms), 2)
                    else:
                        details["elapsed_ms"] = round(float(asr_elapsed_ms), 2)
                details.setdefault("status", "recent_trace")
                for key in (
                    "captured_at_ts",
                    "voice_activity",
                    "asr_voice_activity",
                    "min_asr_dbfs",
                    "speech_window_summary",
                    "recognizer_prewarmed",
                    "recognizer_prewarm_error",
                    "recorded_at_ts",
                ):
                    value = trace_details.get(key)
                    if value is not None and details.get(key) is None:
                        details[key] = value

            if sub_name == "vad":
                vad_elapsed_ms = OperatorConsoleApp._first_numeric_value(
                    trace_details,
                    keys=("vad_elapsed_ms",),
                )
                if isinstance(vad_elapsed_ms, (int, float)):
                    details["elapsed_ms"] = round(float(vad_elapsed_ms), 2)
                details.setdefault("status", "recent_trace")
                for key in ("voice_activity", "streaming_vad", "vad_triggered", "speech_window_summary"):
                    value = trace_details.get(key)
                    if value is not None and details.get(key) is None:
                        details[key] = value

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
                status = str(details.get("status", sub_snapshot.get("health", "unknown")))
                data_status = self._subfunction_data_status(
                    organ_name=organ_name,
                    sub_name=str(sub_name),
                    status=status,
                    details=details,
                )
                entries.append(
                    {
                        "name": sub_name,
                        "health": sub_snapshot.get("health", "unknown"),
                        "data_health": self._data_health(data_status, str(sub_snapshot.get("health", "unknown"))),
                        "data_status": data_status,
                        "driver": details.get("driver", "unknown"),
                        "elapsed_ms": elapsed_ms,
                        "status": status,
                        "error": details.get("error") or details.get("reason") or details.get("stderr", ""),
                        "visual_summary": visual_summary,
                        "probe": probe_details,
                    }
                )
            live_data_count = sum(1 for entry in entries if entry["data_health"] == "healthy")
            waiting_data_count = sum(1 for entry in entries if entry["data_health"] == "degraded")
            data_status = "live" if live_data_count else ("waiting_for_data" if waiting_data_count else "no_data")
            organ_data_status = self._data_health(data_status, str(snapshot.get("health", "unknown")))
            if organ_name == "eye" and data_status == "waiting_for_data" and str(snapshot.get("health", "unknown")) == "healthy":
                organ_data_status = "waiting"
            cards.append(
                {
                    "name": organ_name,
                    "label": self.ORGAN_LABELS.get(organ_name, organ_name.title()),
                    "health": snapshot.get("health", "unknown"),
                    "data_health": organ_data_status,
                    "data_status": data_status,
                    "live_data_subfunctions": live_data_count,
                    "waiting_data_subfunctions": waiting_data_count,
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

    def _build_trace_latency_metrics(self, traces: list[dict[str, object]]) -> list[dict[str, object]]:
        details = self._latest_audio_trace_details(traces)
        if not details:
            return []
        metrics: list[dict[str, object]] = []
        for subfunction, keys in (
            ("capture", ("capture_elapsed_ms", "capture_window_elapsed_ms", "capture_read_elapsed_ms")),
            ("asr", ("asr_elapsed_ms", "asr_decode_elapsed_ms", "asr_infer_elapsed_ms")),
        ):
            elapsed_ms = self._first_numeric_value(details, keys=keys)
            if not isinstance(elapsed_ms, (int, float)):
                continue
            metrics.append(
                {
                    "id": f"ear.{subfunction}.recent",
                    "organ": "ear",
                    "subfunction": subfunction,
                    "driver": "recent_trace",
                    "health": "healthy",
                    "elapsed_ms": round(float(elapsed_ms), 2),
                }
            )
        metrics.sort(key=lambda item: float(item["elapsed_ms"]), reverse=True)
        return metrics

    def _build_dialogue_latency_metrics(
        self,
        *,
        body_snapshot: dict[str, object],
        cognitive_snapshot: dict[str, object],
    ) -> list[dict[str, object]]:
        diagnostics = self._build_dialogue_diagnostics(
            body_snapshot=body_snapshot,
            cognitive_snapshot=cognitive_snapshot,
        )
        stage_latency_ms = diagnostics.get("last_stage_latency_ms", {})
        if not isinstance(stage_latency_ms, dict):
            return []
        metrics: list[dict[str, object]] = []
        for stage_name in ("listen_asr", "think", "speak", "overhead"):
            elapsed_ms = stage_latency_ms.get(stage_name)
            if not isinstance(elapsed_ms, (int, float)):
                continue
            metrics.append(
                {
                    "id": f"voice_dialogue.{stage_name}",
                    "organ": "dialogue",
                    "subfunction": stage_name,
                    "driver": "voice_dialogue_loop",
                    "health": "healthy" if diagnostics.get("running") else "degraded",
                    "elapsed_ms": round(float(elapsed_ms), 2),
                }
            )
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
        return {
            "enabled_capability_count": enabled_capabilities,
            "capability_count": len(capabilities),
            "warning_count": len(warnings),
            "degraded_organ_count": len(degraded_organs),
            "avg_latency_ms": avg_latency,
            "healthy_subfunction_count": runtime_overview["healthy_subfunction_count"],
            "subfunction_count": runtime_overview["subfunction_count"],
            "live_data_subfunction_count": runtime_overview["live_data_subfunction_count"],
            "waiting_data_subfunction_count": runtime_overview["waiting_data_subfunction_count"],
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
        live_data_subfunction_count = sum(int(card.get("live_data_subfunctions", 0)) for card in organ_cards)
        waiting_data_subfunction_count = sum(int(card.get("waiting_data_subfunctions", 0)) for card in organ_cards)
        return {
            "node_id": body_snapshot.get("node_id", "unknown"),
            "degradation_mode": body_snapshot.get("degradation_mode", "unknown"),
            "organ_count": body_snapshot.get("organ_count", len(organ_cards)),
            "recent_event_count": body_snapshot.get("recent_event_count", 0),
            "subfunction_count": subfunction_count,
            "healthy_subfunction_count": healthy_subfunction_count,
            "live_data_subfunction_count": live_data_subfunction_count,
            "waiting_data_subfunction_count": waiting_data_subfunction_count,
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

    def _build_visual_diagnostics(
        self,
        *,
        body_snapshot: dict[str, object],
        organs: dict[str, object],
    ) -> dict[str, object]:
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
        camera_status = str(camera_details.get("status", camera.get("health", "unknown")))
        detection_status = str(detection_details.get("status", detection.get("health", "unknown")))
        vision_service_status = str(
            detection_details.get("service_status")
            or camera_details.get("service_status")
            or camera_status
            or detection_status
        )
        visual_tracking = body_snapshot.get("visual_tracking", {})
        if not isinstance(visual_tracking, dict):
            visual_tracking = {}
        vision_metric_sources = self._nested_dict_sources(detection_details, camera_details, visual_tracking)
        vision_fps = self._first_numeric_from_sources(
            vision_metric_sources,
            keys=("vision_fps", "fps", "current_fps"),
        )
        vision_target_fps = self._first_numeric_from_sources(
            vision_metric_sources,
            keys=("vision_target_fps", "target_fps", "configured_target_fps"),
        )
        vision_loop_interval_s = self._first_numeric_from_sources(
            vision_metric_sources,
            keys=("vision_loop_interval_s", "interval_s", "configured_interval_s", "loop_interval_s"),
        )
        if vision_target_fps is None and isinstance(vision_loop_interval_s, (int, float)) and vision_loop_interval_s > 0:
            vision_target_fps = round(1.0 / float(vision_loop_interval_s), 4)
        if vision_loop_interval_s is None and isinstance(vision_target_fps, (int, float)) and vision_target_fps > 0:
            vision_loop_interval_s = round(1.0 / float(vision_target_fps), 4)
        explicit_frame_age_s = self._first_numeric_from_sources(
            vision_metric_sources,
            keys=("vision_frame_age_s", "frame_age_s", "last_frame_age_s", "last_frame_age"),
        )
        frame_age_s = explicit_frame_age_s if explicit_frame_age_s is not None else self._age_seconds(frame_captured_at_ts)
        vision_frame_status = str(
            self._first_present(
                *vision_metric_sources,
                keys=("vision_frame_status", "frame_status", "freshness_status"),
                default="",
            )
            or ""
        )
        registered_identity = body_snapshot.get("identity_registry", {})
        if not isinstance(registered_identity, dict):
            registered_identity = {}
        tracking_target = visual_tracking.get("target")
        if isinstance(tracking_target, dict):
            tracking_target = dict(tracking_target)
        else:
            tracking_target = None
        tracking_decision = visual_tracking.get("tracking_decision", {})
        if not isinstance(tracking_decision, dict):
            tracking_decision = {}
        tracking_target_center_x = visual_tracking.get("tracking_target_center_x")
        if tracking_target_center_x is None and isinstance(tracking_target, dict):
            tracking_target_center_x = tracking_target.get("target_x")
        tracking_target_error_x = visual_tracking.get("tracking_target_error_x")
        if tracking_target_error_x is None:
            tracking_target_error_x = self._target_error_x(tracking_target_center_x)
        tracking_suppressed_reason = str(
            visual_tracking.get("tracking_suppressed_reason") or tracking_decision.get("reason") or ""
        )
        recognized_identity: dict[str, object] = {}
        identity_summary = identity_details.get("identity_summary", "waiting for identity data")
        if registered_identity.get("registered"):
            display_name = str(registered_identity.get("display_name", "known person") or "known person")
            actor_id = str(registered_identity.get("actor_id", "") or "")
            recognized_identity = {
                "actor_id": actor_id,
                "display_name": display_name,
                "source": registered_identity.get("source", "session"),
                "registered_at_ts": registered_identity.get("registered_at_ts"),
            }
            identity_summary = f"recognized {display_name} (session registration)"
            if tracking_target is not None and isinstance(tracking_target.get("bbox"), dict):
                tracking_target["identity"] = display_name
                tracking_target["actor_id"] = actor_id
                tracking_target["registered_identity"] = True
            registered_target = registered_identity.get("target")
            if isinstance(registered_target, dict):
                registered_candidate = {
                    "candidate_id": "session_registered_target",
                    "identity": display_name,
                    "actor_id": actor_id,
                    "score": registered_target.get("score", 1.0),
                    "bbox": registered_target.get("bbox", {}),
                    "source": "session_registration",
                }
                identity_candidates = [registered_candidate, *identity_candidates]
        data_status = "live" if frame_path else "waiting_for_frame"
        if vision_service_status == "sleeping" or camera_status == "sleeping" or detection_status == "sleeping":
            data_status = "sleeping"
        data_health = "degraded" if data_status in {"waiting_for_data", "waiting_for_frame"} else "healthy"
        if data_status == "waiting_for_frame":
            data_health = "waiting"
        elif data_status == "live":
            data_health = "healthy"
        elif data_status == "sleeping":
            data_health = "healthy"
        if not vision_frame_status:
            vision_frame_status = "live" if data_status == "live" else data_status

        top_detection = detection_details.get("top_detection")
        top_detection_bbox = top_detection.get("bbox") if isinstance(top_detection, dict) else None
        return {
            "enabled": bool(frame_path or detections or identity_candidates or registered_identity.get("registered")),
            "data_health": data_health,
            "data_status": data_status,
            "frame_available": bool(frame_path),
            "frame_url": "/vision/latest.jpg" if frame_path else None,
            "frame_captured_at_ts": frame_captured_at_ts,
            "frame_status": vision_frame_status,
            "camera_health": camera.get("health", "unknown"),
            "detection_health": detection.get("health", "unknown"),
            "identity_health": identity.get("health", "unknown"),
            "detection_status": detection_details.get("status", detection.get("health", "unknown")),
            "identity_status": identity_details.get("status", identity.get("health", "unknown")),
            "detection_count": len(detections),
            "detections": detections,
            "identity_candidates": identity_candidates,
            "registered_identity": registered_identity,
            "recognized_identity": recognized_identity,
            "scene_summary": detection_details.get("scene_summary", "waiting for camera/detection data"),
            "identity_summary": identity_summary,
            "scene_labels": detection_details.get("scene_labels", []),
            "top_detection": detection_details.get("top_detection"),
            "top_detection_bbox": top_detection_bbox,
            "tracking_source": visual_tracking.get("source", ""),
            "backend": detection_details.get("backend") or camera_details.get("backend"),
            "vision_service_status": vision_service_status,
            "state_path": detection_details.get("state_path") or camera_details.get("state_path"),
            "state_age_s": detection_details.get("state_age_s", camera_details.get("state_age_s")),
            "state_updated_at_ts": detection_details.get("state_updated_at_ts", camera_details.get("state_updated_at_ts")),
            "frame_age_s": frame_age_s,
            "vision_fps": vision_fps,
            "vision_target_fps": vision_target_fps,
            "vision_loop_interval_s": vision_loop_interval_s,
            "vision_frame_age_s": frame_age_s,
            "vision_frame_status": vision_frame_status,
            "tracking_status": visual_tracking.get("status", "idle"),
            "tracking_updated_at_ts": visual_tracking.get("updated_at_ts"),
            "tracking_age_s": self._age_seconds(visual_tracking.get("updated_at_ts")),
            "tracking_running": bool(visual_tracking.get("running", False)),
            "tracking_target": tracking_target,
            "tracking_target_center_x": tracking_target_center_x,
            "tracking_target_error_x": tracking_target_error_x,
            "tracking_decision": tracking_decision,
            "tracking_suppressed_reason": tracking_suppressed_reason,
            "tracking_miss_count": int(visual_tracking.get("miss_count", 0) or 0),
            "tracking_last_outcome_status": visual_tracking.get("last_outcome_status"),
            "tracking_last_error": str(visual_tracking.get("last_error", "") or ""),
        }

    def _build_audio_diagnostics(self, organs: dict[str, object], *, traces: list[dict[str, object]] | None = None) -> dict[str, object]:
        ear = organs.get("ear", {})
        if not isinstance(ear, dict):
            ear = {}
        subfunctions = ear.get("subfunctions", {})
        if not isinstance(subfunctions, dict):
            subfunctions = {}
        capture = subfunctions.get("capture", {})
        vad = subfunctions.get("vad", {})
        asr = subfunctions.get("asr", {})
        if not isinstance(capture, dict):
            capture = {}
        if not isinstance(vad, dict):
            vad = {}
        if not isinstance(asr, dict):
            asr = {}
        capture_details = dict(capture.get("details", {})) if isinstance(capture.get("details", {}), dict) else {}
        vad_details = dict(vad.get("details", {})) if isinstance(vad.get("details", {}), dict) else {}
        asr_details = dict(asr.get("details", {})) if isinstance(asr.get("details", {}), dict) else {}
        trace_details = self._latest_audio_trace_details(traces or [])
        if trace_details and capture_details.get("dbfs") is None:
            capture_elapsed_ms = self._first_numeric_value(
                trace_details,
                keys=("capture_elapsed_ms", "capture_window_elapsed_ms", "capture_read_elapsed_ms", "capture_latency_ms"),
            )
            capture_details.update(
                {
                    "driver": capture_details.get("driver", "recent_trace"),
                    "status": "recent_trace",
                    "capture_device": trace_details.get("capture_device"),
                    "sample_rate": trace_details.get("sample_rate"),
                    "channels": trace_details.get("channels"),
                    "chunk_count": trace_details.get("chunk_count"),
                    "payload_bytes": trace_details.get("payload_bytes"),
                    "dbfs": trace_details.get("dbfs"),
                    "rms_level": trace_details.get("rms_level"),
                    "peak_level": trace_details.get("peak_level"),
                    "voice_activity": (
                        trace_details.get("voice_activity")
                        if "voice_activity" in trace_details
                        else bool(trace_details.get("dbfs"))
                    ),
                    "streaming_vad": trace_details.get("streaming_vad"),
                    "vad_triggered": trace_details.get("vad_triggered"),
                    "vad_elapsed_ms": trace_details.get("vad_elapsed_ms"),
                    "captured_at_ts": trace_details.get("captured_at_ts") or trace_details.get("recorded_at_ts"),
                    "elapsed_ms": capture_elapsed_ms,
                }
            )
        if trace_details and not asr_details.get("transcript"):
            transcript_from_trace = str(trace_details.get("text", "") or "")
            asr_elapsed_ms = self._first_numeric_value(
                trace_details,
                keys=("asr_elapsed_ms", "asr_decode_elapsed_ms", "asr_infer_elapsed_ms"),
            )
            asr_decode_elapsed_ms = self._first_numeric_value(
                trace_details,
                keys=("asr_decode_elapsed_ms", "asr_infer_elapsed_ms"),
            )
            asr_details.update(
                {
                    "driver": asr_details.get("driver", "recent_trace"),
                    "status": trace_details.get("asr_status") or ("transcribed" if transcript_from_trace else "no_transcript"),
                    "transcript": transcript_from_trace,
                    "voice_activity": (
                        trace_details.get("voice_activity")
                        if "voice_activity" in trace_details
                        else bool(trace_details.get("dbfs"))
                    ),
                    "asr_voice_activity": trace_details.get("asr_voice_activity"),
                    "min_asr_dbfs": trace_details.get("min_asr_dbfs"),
                    "recognizer_prewarmed": trace_details.get("recognizer_prewarmed"),
                    "recognizer_prewarm_error": trace_details.get("recognizer_prewarm_error"),
                    "speech_window_summary": trace_details.get("speech_window_summary"),
                    "captured_at_ts": trace_details.get("captured_at_ts") or trace_details.get("recorded_at_ts"),
                    "asr_elapsed_ms": asr_elapsed_ms,
                    "asr_decode_elapsed_ms": asr_decode_elapsed_ms,
                }
            )
            asr_details["elapsed_ms"] = asr_elapsed_ms
        elif trace_details:
            asr_elapsed_ms = self._first_numeric_value(
                trace_details,
                keys=("asr_elapsed_ms", "asr_decode_elapsed_ms", "asr_infer_elapsed_ms"),
            )
            asr_decode_elapsed_ms = self._first_numeric_value(
                trace_details,
                keys=("asr_decode_elapsed_ms", "asr_infer_elapsed_ms"),
            )
            asr_details.update(
                {
                    "asr_elapsed_ms": asr_elapsed_ms,
                    "asr_decode_elapsed_ms": asr_decode_elapsed_ms,
                }
            )
            if asr_details.get("elapsed_ms") is None and isinstance(asr_elapsed_ms, (int, float)):
                asr_details["elapsed_ms"] = asr_elapsed_ms
        transcript = str(asr_details.get("transcript", "") or "")
        return {
            "enabled": bool(capture_details or asr_details),
            "capture_health": capture.get("health", "unknown"),
            "vad_health": vad.get("health", "unknown"),
            "asr_health": asr.get("health", "unknown"),
            "capture_status": capture_details.get("status", capture.get("health", "unknown")),
            "vad_status": vad_details.get("status", vad.get("health", "unknown")),
            "asr_status": asr_details.get("status", asr.get("health", "unknown")),
            "capture_device": capture_details.get("capture_device") or capture_details.get("device"),
            "sample_rate": capture_details.get("sample_rate"),
            "channels": capture_details.get("channels"),
            "chunk_count": capture_details.get("chunk_count"),
            "payload_bytes": capture_details.get("payload_bytes"),
            "dbfs": capture_details.get("dbfs"),
            "rms_level": capture_details.get("rms_level"),
            "peak_level": capture_details.get("peak_level"),
            "voice_activity": bool(asr_details.get("voice_activity", capture_details.get("voice_activity"))),
            "streaming_vad": capture_details.get("streaming_vad") or asr_details.get("streaming_vad"),
            "vad_triggered": capture_details.get("vad_triggered") if capture_details.get("vad_triggered") is not None else asr_details.get("vad_triggered"),
            "vad_elapsed_ms": capture_details.get("vad_elapsed_ms") or asr_details.get("vad_elapsed_ms"),
            "recognizer_prewarmed": asr_details.get("recognizer_prewarmed"),
            "recognizer_prewarm_error": asr_details.get("recognizer_prewarm_error"),
            "captured_at_ts": asr_details.get("captured_at_ts") or capture_details.get("captured_at_ts"),
            "transcript": transcript,
            "transcript_char_count": len(transcript),
            "speech_window_summary": asr_details.get("speech_window_summary")
            or vad_details.get("speech_window_summary")
            or capture_details.get("speech_window_summary"),
            "capture_elapsed_ms": capture_details.get("elapsed_ms"),
            "asr_elapsed_ms": asr_details.get("asr_elapsed_ms", asr_details.get("elapsed_ms")),
            "asr_decode_elapsed_ms": asr_details.get("asr_decode_elapsed_ms"),
        }

    @staticmethod
    def _latest_audio_trace_details(traces: list[dict[str, object]]) -> dict[str, object]:
        for trace in reversed(traces):
            if not isinstance(trace, dict) or trace.get("kind") != "audio_transcript_final":
                continue
            details = trace.get("details", {})
            if not isinstance(details, dict):
                continue
            merged = dict(details)
            if "recorded_at_ts" in trace:
                merged.setdefault("recorded_at_ts", trace.get("recorded_at_ts"))
            return merged
        return {}

    @staticmethod
    def _first_numeric_value(value: dict[str, object], *, keys: tuple[str, ...]) -> float | int | None:
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, (int, float)):
                return candidate
        return None

    @classmethod
    def _first_numeric_from_sources(
        cls,
        sources: list[dict[str, object]],
        *,
        keys: tuple[str, ...],
    ) -> float | int | None:
        for source in sources:
            value = cls._first_numeric_value(source, keys=keys)
            if value is not None:
                return value
        return None

    @staticmethod
    def _nested_dict_sources(*values: object) -> list[dict[str, object]]:
        sources: list[dict[str, object]] = []

        def add(value: object, *, depth: int) -> None:
            if depth > 3 or not isinstance(value, dict):
                return
            source = dict(value)
            sources.append(source)
            for nested in value.values():
                if isinstance(nested, dict):
                    add(nested, depth=depth + 1)

        for value in values:
            add(value, depth=0)
        return sources

    @staticmethod
    def _age_seconds(timestamp: object) -> float | None:
        if not isinstance(timestamp, (int, float)):
            return None
        return round(max(0.0, time.time() - float(timestamp)), 2)

    @staticmethod
    def _build_dialogue_diagnostics(
        *,
        body_snapshot: dict[str, object],
        cognitive_snapshot: dict[str, object],
    ) -> dict[str, object]:
        loop = body_snapshot.get("voice_dialogue", {})
        if not isinstance(loop, dict):
            loop = {}
        phase_started_at_ts = loop.get("phase_started_at_ts")
        if isinstance(phase_started_at_ts, (int, float)):
            current_phase_elapsed_s = round(time.time() - float(phase_started_at_ts), 2)
        else:
            current_phase_elapsed_s = loop.get("current_phase_elapsed_s", 0.0)
        last_latency_s = loop.get("last_latency_s", {})
        if not isinstance(last_latency_s, dict):
            last_latency_s = {}
        last_stage_latency_ms = loop.get("last_stage_latency_ms", {})
        if not isinstance(last_stage_latency_ms, dict) or not last_stage_latency_ms:
            last_stage_latency_ms = OperatorConsoleApp._derive_stage_latency_ms(last_latency_s)
        voice_chain_readiness = OperatorConsoleApp._build_voice_chain_readiness(loop)
        return {
            "enabled": bool(loop.get("enabled")),
            "running": bool(loop.get("running")),
            "conversation_active": bool(loop.get("conversation_active")),
            "wake_word": str(loop.get("wake_word", "")),
            "sleep_word": str(loop.get("sleep_word", "")),
            "phase": loop.get("phase", "idle"),
            "phase_started_at_ts": loop.get("phase_started_at_ts"),
            "current_phase_elapsed_s": current_phase_elapsed_s,
            "last_status": loop.get("last_status", "idle"),
            "turn_count": loop.get("turn_count", 0),
            "last_transcript": loop.get("last_transcript", ""),
            "last_reply": loop.get("last_reply") or cognitive_snapshot.get("last_reply", ""),
            "last_error": loop.get("last_error", ""),
            "last_latency_s": last_latency_s,
            "last_stage_latency_ms": last_stage_latency_ms,
            "last_bottleneck_stage": loop.get("last_bottleneck_stage") or OperatorConsoleApp._bottleneck_stage(last_stage_latency_ms),
            "last_bottleneck_ms": loop.get("last_bottleneck_ms", OperatorConsoleApp._bottleneck_ms(last_stage_latency_ms)),
            "last_completed_turn": loop.get("last_completed_turn", {}),
            "updated_at_ts": loop.get("updated_at_ts"),
            "learning_decision": cognitive_snapshot.get("learning_decision", ""),
            "last_review": cognitive_snapshot.get("last_review", {}),
            "last_llm_status": cognitive_snapshot.get("last_llm_status", {}),
            "voice_chain_readiness": voice_chain_readiness,
        }

    @classmethod
    def _build_memory_diagnostics(
        cls,
        cognitive_snapshot: dict[str, object],
        *,
        memory_trace_panel: dict[str, object] | None = None,
    ) -> dict[str, object]:
        diagnostics = cognitive_snapshot.get("memory_diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        if memory_trace_panel is None:
            memory_trace_panel = cls._build_memory_trace_panel(cognitive_snapshot)
        latest_trace = memory_trace_panel.get("latest", {}) if isinstance(memory_trace_panel, dict) else {}
        if not isinstance(latest_trace, dict):
            latest_trace = {}
        recall = cls._first_dict(
            diagnostics,
            cognitive_snapshot,
            keys=("recall", "last_recall", "memory_recall", "last_memory_recall"),
        )
        query = cls._first_present(
            diagnostics,
            recall,
            cognitive_snapshot,
            keys=("last_memory_query", "memory_query", "query", "last_query"),
            default="",
        )
        task_context = cls._first_present(
            diagnostics,
            recall,
            cognitive_snapshot,
            keys=("task_context", "memory_task_context", "last_task_context"),
            default={},
        )
        recall_profile = cls._first_present(
            diagnostics,
            recall,
            cognitive_snapshot,
            keys=("recall_profile", "profile", "memory_profile"),
            default="",
        )
        allowed_sources = cls._as_list(
            cls._first_present(
                diagnostics,
                recall,
                cognitive_snapshot,
                keys=("allowed_sources", "source_allowlist", "memory_allowed_sources"),
                default=[],
            )
        )
        blocked_sources = cls._as_list(
            cls._first_present(
                diagnostics,
                recall,
                cognitive_snapshot,
                keys=("blocked_sources", "source_blocklist", "memory_blocked_sources"),
                default=[],
            )
        )
        selected_records = cls._as_record_list(
            cls._first_present(
                diagnostics,
                recall,
                cognitive_snapshot,
                keys=("selected_records", "recall_selected_records", "selected", "records", "relevant_memories"),
                default=[],
            )
        )
        if not selected_records:
            selected_records = cls._as_record_list(latest_trace.get("selected_records", []))
        source_composition = cls._normalize_source_composition(
            cls._first_present(
                diagnostics,
                recall,
                cognitive_snapshot,
                keys=("source_composition", "selected_source_composition", "sources"),
                default={},
            ),
            selected_records=selected_records,
        )
        if not source_composition and isinstance(latest_trace.get("source_composition"), dict):
            source_composition = cls._normalize_source_composition(
                latest_trace.get("source_composition"),
                selected_records=selected_records,
            )
        selected_count = cls._first_present(
            diagnostics,
            recall,
            cognitive_snapshot,
            keys=("selected_count", "recall_selected_count"),
            default=latest_trace.get("selected_count", len(selected_records)),
        )
        if not isinstance(selected_count, int):
            selected_count = len(selected_records)
        last_writeback = cls._normalize_writeback(
            cls._first_present(
                diagnostics,
                cognitive_snapshot,
                keys=("last_writeback", "last_memory_writeback", "writeback_status", "memory_writeback"),
                default={},
            )
        )
        if not last_writeback and isinstance(latest_trace.get("latest_writeback"), dict):
            last_writeback = cls._normalize_writeback(latest_trace.get("latest_writeback"))
        memory_trace_count = int(memory_trace_panel.get("count", 0)) if isinstance(memory_trace_panel, dict) else 0
        return {
            "enabled": bool(diagnostics or recall or query or task_context or selected_records or source_composition or last_writeback or memory_trace_count),
            "provider": cls._first_present(
                diagnostics,
                cognitive_snapshot,
                keys=("provider", "memory_provider"),
                default="",
            ),
            "endpoint": cls._first_present(
                diagnostics,
                cognitive_snapshot,
                keys=("endpoint", "memory_endpoint"),
                default="",
            ),
            "channel_owner": cls._first_present(
                diagnostics,
                cognitive_snapshot,
                keys=("channel_owner",),
                default="eibrain",
            ),
            "agent_owner": cls._first_present(
                diagnostics,
                cognitive_snapshot,
                keys=("agent_owner",),
                default="eibrain",
            ),
            "memory_owner": cls._first_present(
                diagnostics,
                cognitive_snapshot,
                keys=("memory_owner",),
                default="local_in_memory",
            ),
            "last_query": query,
            "last_memory_query": query,
            "task_context": task_context if isinstance(task_context, dict) else {},
            "recall_profile": recall_profile,
            "allowed_sources": allowed_sources,
            "blocked_sources": blocked_sources,
            "selected_count": selected_count,
            "selected_records": selected_records,
            "source_composition": source_composition,
            "last_writeback": last_writeback,
            "memory_trace_count": memory_trace_count,
            "latest_trace_round_id": latest_trace.get("round_id", ""),
            "latest_trace_status": latest_trace.get("status", ""),
            "latest_trace": latest_trace,
        }

    @classmethod
    def _build_memory_monitor_snapshot(
        cls,
        *,
        body_snapshot: dict[str, object],
        cognitive_snapshot: dict[str, object],
    ) -> dict[str, object]:
        merged = dict(cognitive_snapshot)
        current = dict(merged.get("current", {})) if isinstance(merged.get("current"), dict) else {}
        scheduler = dict(merged.get("scheduler", {})) if isinstance(merged.get("scheduler"), dict) else {}
        merged_traces = cls._extract_memory_traces(merged)
        scheduler_count = (
            int(scheduler.get("memory_trace_count", 0))
            if isinstance(scheduler.get("memory_trace_count"), int)
            else 0
        )

        for source in cls._live_voice_memory_sources(body_snapshot):
            for trace in cls._extract_memory_traces(source):
                if trace not in merged_traces:
                    merged_traces.append(trace)
            source_scheduler = source.get("scheduler", {})
            if not isinstance(source_scheduler, dict):
                source_scheduler = source
            source_count = source_scheduler.get("memory_trace_count")
            if isinstance(source_count, int):
                scheduler_count = max(scheduler_count, source_count)

        if merged_traces:
            current["memory_traces"] = merged_traces
            merged["current"] = current
        if scheduler_count:
            scheduler["memory_trace_count"] = max(scheduler_count, len(merged_traces))
            merged["scheduler"] = scheduler
        return merged

    @classmethod
    def _live_voice_memory_sources(cls, body_snapshot: dict[str, object]) -> list[dict[str, object]]:
        voice_dialogue = body_snapshot.get("voice_dialogue", {})
        if not isinstance(voice_dialogue, dict):
            return []
        sources: list[dict[str, object]] = []
        for key in (
            "scheduler_state",
            "scheduler",
            "realtime_cognition",
            "realtime_session",
            "latest_realtime_session",
        ):
            value = voice_dialogue.get(key)
            if isinstance(value, dict):
                sources.append(value)
        return sources

    @classmethod
    def _build_memory_trace_panel(cls, cognitive_snapshot: dict[str, object]) -> dict[str, object]:
        traces = [cls._normalize_memory_trace(trace) for trace in cls._extract_memory_traces(cognitive_snapshot)]
        traces = [trace for trace in traces if trace]
        scheduler = cognitive_snapshot.get("scheduler", {})
        scheduler_count = 0
        if isinstance(scheduler, dict) and isinstance(scheduler.get("memory_trace_count"), int):
            scheduler_count = int(scheduler["memory_trace_count"])
        count = max(len(traces), scheduler_count)
        latest = traces[-1] if traces else {}
        recent = list(reversed(traces[-5:]))
        return {
            "enabled": bool(count),
            "count": count,
            "latest": latest,
            "items": recent,
        }

    @classmethod
    def _extract_memory_traces(cls, cognitive_snapshot: dict[str, object]) -> list[dict[str, object]]:
        containers: list[dict[str, object]] = [cognitive_snapshot]
        for key in ("current", "memory_diagnostics"):
            value = cognitive_snapshot.get(key, {})
            if isinstance(value, dict):
                containers.append(value)
        traces: list[dict[str, object]] = []
        for container in containers:
            for key in ("memory_traces", "closed_loop_traces", "memory_trace_history"):
                value = container.get(key)
                if isinstance(value, list):
                    traces.extend(dict(item) for item in value if isinstance(item, dict))
        return traces

    @classmethod
    def _normalize_memory_trace(cls, trace: dict[str, object]) -> dict[str, object]:
        recall = trace.get("recall", {})
        if not isinstance(recall, dict):
            recall = {}
        writeback = trace.get("writeback", {})
        if not isinstance(writeback, dict):
            writeback = {}
        recall_items = [cls._normalize_recall_trace_item(item) for item in cls._as_list(recall.get("items", []))]
        recall_items = [item for item in recall_items if item]
        writeback_items = [cls._normalize_writeback_trace_item(item) for item in cls._as_list(writeback.get("items", []))]
        writeback_items = [item for item in writeback_items if item]
        selected_records: list[dict[str, object]] = []
        source_composition: dict[str, int] = {}
        for item in recall_items:
            selected_records.extend(cls._as_record_list(item.get("selected_records", [])))
            for source, count in cls._normalize_source_composition(
                item.get("source_composition", {}),
                selected_records=[],
            ).items():
                source_composition[source] = source_composition.get(source, 0) + count
        if not source_composition:
            source_composition = cls._normalize_source_composition({}, selected_records=selected_records)
        errors = cls._as_record_list(trace.get("errors", []))
        recall_count = cls._trace_count(recall, fallback=len(recall_items))
        writeback_count = cls._trace_count(writeback, fallback=sum(1 for item in writeback_items if item.get("status") != "skipped"))
        error_count = len(errors)
        latest_writeback = writeback_items[0] if writeback_items else {}
        return {
            "schema": trace.get("schema", ""),
            "round_id": trace.get("round_id", ""),
            "session_id": trace.get("session_id", ""),
            "actor_id": trace.get("actor_id", ""),
            "status": "error" if error_count else ("ok" if recall_count or writeback_count else "waiting"),
            "recall_count": recall_count,
            "writeback_count": writeback_count,
            "error_count": error_count,
            "selected_count": sum(int(item.get("selected_count", 0)) for item in recall_items if isinstance(item.get("selected_count"), int)),
            "selected_records": selected_records,
            "source_composition": source_composition,
            "recall_items": recall_items,
            "writeback_items": writeback_items,
            "latest_writeback": latest_writeback,
            "errors": errors,
        }

    @staticmethod
    def _trace_count(section: dict[str, object], *, fallback: int) -> int:
        value = section.get("count")
        return int(value) if isinstance(value, int) else fallback

    @classmethod
    def _normalize_recall_trace_item(cls, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        selected_records = cls._as_record_list(value.get("selected_records", []))
        return {
            "query": value.get("query", ""),
            "summary": value.get("summary", ""),
            "selected_count": value.get("selected_count", len(selected_records)),
            "selected_records": selected_records,
            "source_composition": cls._normalize_source_composition(
                value.get("source_composition", {}),
                selected_records=selected_records,
            ),
        }

    @classmethod
    def _normalize_writeback_trace_item(cls, value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        diagnostics = value.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        normalized = cls._normalize_writeback({**diagnostics, **value})
        normalized["record_id"] = value.get("record_id") or diagnostics.get("record_id") or normalized.get("record_id", "")
        normalized["summary"] = value.get("summary", normalized.get("summary", ""))
        return normalized

    @classmethod
    def _build_neck_control_diagnostics(cls, *, body_snapshot: dict[str, object]) -> dict[str, object]:
        neck_control = body_snapshot.get("neck_control", {})
        if not isinstance(neck_control, dict):
            neck_control = {}
        if not neck_control:
            for source in (
                body_snapshot.get("body_state", {}),
                body_snapshot,
            ):
                if not isinstance(source, dict):
                    continue
                organs = source.get("organs", {})
                if not isinstance(organs, dict):
                    continue
                neck = organs.get("neck", {})
                if not isinstance(neck, dict):
                    continue
                subfunctions = neck.get("subfunctions", {})
                if not isinstance(subfunctions, dict):
                    continue
                for subfunction in subfunctions.values():
                    if not isinstance(subfunction, dict):
                        continue
                    details = subfunction.get("details", {})
                    if isinstance(details, dict) and isinstance(details.get("neck_control"), dict):
                        neck_control = details["neck_control"]
                        break
                if neck_control:
                    break
        active_intent = neck_control.get("active_intent", {})
        if not isinstance(active_intent, dict):
            active_intent = {}
        last_command_status = neck_control.get("last_command_status", {})
        if isinstance(last_command_status, str):
            last_command_status = {"status": last_command_status}
        elif not isinstance(last_command_status, dict):
            last_command_status = {}
        capabilities = body_snapshot.get("capabilities", {})
        can_orient_head = isinstance(capabilities, dict) and bool(capabilities.get("can_orient_head"))
        return {
            "enabled": bool(neck_control) or can_orient_head,
            "state": neck_control.get("state", "unavailable"),
            "active_intent": str(active_intent.get("intent") or active_intent.get("target_name") or ""),
            "active_source": str(active_intent.get("source", "")),
            "desired_angle": neck_control.get("desired_angle", 0.0),
            "last_angle": neck_control.get("last_angle", 0.0),
            "suppressed_reason": str(neck_control.get("suppressed_reason", "")),
            "last_command_status": last_command_status,
            "last_command_status_label": str(last_command_status.get("status", "")),
            "intent_count": int(neck_control.get("intent_count", 0)),
            "pan_motion_proof": cls._load_pan_motion_proof(),
        }

    @classmethod
    def _load_pan_motion_proof(cls) -> dict[str, object]:
        path = Path(cls.PAN_MOTION_PROOF_PATH)
        if not path.exists():
            return {"status": "missing", "verified": False, "path": str(path)}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"status": "error", "verified": False, "path": str(path), "error": str(exc)}
        if not isinstance(payload, dict):
            return {"status": "invalid", "verified": False, "path": str(path)}
        proof = dict(payload)
        proof.setdefault("status", "available")
        proof.setdefault("verified", bool(proof.get("status") == "verified"))
        proof["path"] = str(path)
        try:
            proof["age_s"] = round(time.time() - path.stat().st_mtime, 2)
        except OSError:
            pass
        return proof

    @staticmethod
    def _target_error_x(value: object) -> float | None:
        try:
            return round(float(value) - 0.5, 4)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _derive_stage_latency_ms(last_latency_s: dict[str, object]) -> dict[str, float]:
        stage_latency_ms: dict[str, float] = {}
        for stage_name in ("listen_asr", "think", "speak", "total"):
            value = last_latency_s.get(stage_name)
            if isinstance(value, (int, float)):
                stage_latency_ms[stage_name] = round(float(value) * 1000, 2)
        if stage_latency_ms:
            stage_latency_ms["overhead"] = round(
                max(
                    0.0,
                    float(stage_latency_ms.get("total", 0.0))
                    - float(stage_latency_ms.get("listen_asr", 0.0))
                    - float(stage_latency_ms.get("think", 0.0))
                    - float(stage_latency_ms.get("speak", 0.0)),
                ),
                2,
            )
        return stage_latency_ms

    @staticmethod
    def _bottleneck_stage(stage_latency_ms: dict[str, object]) -> str:
        candidates = {
            key: float(value)
            for key, value in stage_latency_ms.items()
            if key != "total" and isinstance(value, (int, float))
        }
        if not candidates:
            return ""
        return max(candidates, key=candidates.get)

    @classmethod
    def _bottleneck_ms(cls, stage_latency_ms: dict[str, object]) -> float | None:
        stage = cls._bottleneck_stage(stage_latency_ms)
        if not stage:
            return None
        value = stage_latency_ms.get(stage)
        return round(float(value), 2) if isinstance(value, (int, float)) else None

    @staticmethod
    def _build_voice_chain_readiness(loop: dict[str, object]) -> dict[str, object]:
        explicit = loop.get("voice_chain_readiness")
        benchmark = loop.get("voice_chain_benchmark")
        return build_voice_chain_readiness(
            explicit=explicit if isinstance(explicit, dict) else None,
            benchmark=benchmark if isinstance(benchmark, dict) else None,
        )

    @staticmethod
    def _first_dict(*containers: dict[str, object], keys: tuple[str, ...]) -> dict[str, object]:
        value = OperatorConsoleApp._first_present(*containers, keys=keys, default={})
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _first_present(*containers: dict[str, object], keys: tuple[str, ...], default: object) -> object:
        for container in containers:
            if not isinstance(container, dict):
                continue
            for key in keys:
                if key in container and container[key] is not None:
                    return container[key]
        return default

    @staticmethod
    def _as_list(value: object) -> list[object]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return sorted(value)
        if value in (None, ""):
            return []
        return [value]

    @staticmethod
    def _as_record_list(value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        records: list[dict[str, object]] = []
        for index, item in enumerate(value):
            if isinstance(item, dict):
                records.append(dict(item))
            elif item not in (None, ""):
                records.append({"id": f"memory-{index + 1}", "summary": str(item), "source": "unknown"})
        return records

    @staticmethod
    def _normalize_source_composition(
        value: object,
        *,
        selected_records: list[dict[str, object]],
    ) -> dict[str, int]:
        composition: dict[str, int] = {}
        if isinstance(value, dict):
            for source, count in value.items():
                if isinstance(count, (int, float)):
                    composition[str(source)] = int(count)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    source = str(item.get("source") or item.get("name") or "unknown")
                    count = item.get("count", 0)
                    if isinstance(count, (int, float)):
                        composition[source] = composition.get(source, 0) + int(count)
        if composition:
            return composition
        for record in selected_records:
            source = str(record.get("source") or record.get("memory_source") or record.get("type") or "unknown")
            composition[source] = composition.get(source, 0) + 1
        return composition

    @staticmethod
    def _normalize_writeback(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        if not any(
            value.get(key)
            for key in ("source", "type", "memory_type", "status", "summary", "record_id", "modality", "organ")
        ):
            return {}
        memory_type = value.get("memory_type") or value.get("type", "")
        return {
            "source": value.get("source", ""),
            "type": memory_type,
            "memory_type": memory_type,
            "modality": value.get("modality", ""),
            "organ": value.get("organ", ""),
            "status": value.get("status", ""),
            "summary": value.get("summary", ""),
            "record_id": value.get("record_id", ""),
            "updated_at_ts": value.get("updated_at_ts") or value.get("written_at_ts"),
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
    def _data_health(data_status: str, fallback_health: str = "unknown") -> str:
        if data_status in {"live", "recent_trace", "played", "planned"}:
            return "healthy"
        if fallback_health == "unavailable":
            return "unavailable"
        if data_status in {"waiting_for_data", "waiting_for_frame", "waiting_for_action", "waiting_for_target"}:
            return "degraded"
        return "degraded"

    @staticmethod
    def _subfunction_data_status(
        *,
        organ_name: str,
        sub_name: str,
        status: str,
        details: dict[str, object],
    ) -> str:
        now_ts = time.time()
        if status == "live_probe_skipped":
            return "waiting_for_data"
        if organ_name == "eye":
            if details.get("frame_path") or details.get("frame_captured_at_ts"):
                return "live"
            return "waiting_for_frame"
        if organ_name == "mouth":
            if details.get("played_at_ts"):
                return "played"
            if details.get("planned_at_ts"):
                return "planned"
            return "waiting_for_action"
        if organ_name == "neck":
            if details.get("tracked_at_ts") or details.get("target_angle") is not None:
                return "live"
            return "waiting_for_target"
        if organ_name == "ear":
            captured_at_ts = details.get("captured_at_ts")
            if (
                status
                in {"listening", "listening_loop", "no_transcript", "short_transcript_ignored", "transcribed", "below_asr_threshold", "silence"}
            ):
                return "live"
            if (
                isinstance(captured_at_ts, (int, float))
                and now_ts - float(captured_at_ts) <= 8.0
            ):
                return "live"
            if details.get("transcript") or details.get("payload_bytes") or details.get("voice_activity"):
                return "live"
            if details.get("streaming_vad") or details.get("vad_triggered"):
                return "live"
        if details.get("elapsed_ms") is not None:
            return "live"
        return status or "waiting_for_data"

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

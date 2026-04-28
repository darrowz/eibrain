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
        self._annotate_live_voice_loop(organs, body_snapshot=body_snapshot)
        organ_cards = self._build_organ_cards(organs)
        latency_metrics = self._build_latency_metrics(organs)
        if not latency_metrics:
            latency_metrics = self._build_trace_latency_metrics(traces)
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
        memory_diagnostics = self._build_memory_diagnostics(cognitive_snapshot)
        neck_control_diagnostics = self._build_neck_control_diagnostics(body_snapshot)
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
            "memory_diagnostics": memory_diagnostics,
            "neck_control_diagnostics": neck_control_diagnostics,
            "organ_cards": organ_cards,
            "latency_metrics": latency_metrics,
            "event_breakdown": self._build_event_breakdown(traces),
            "body": body_snapshot,
            "cognition": cognitive_snapshot,
            "recent_traces": traces[-5:],
        }

    @staticmethod
    def _annotate_live_voice_loop(organs: dict[str, object], *, body_snapshot: dict[str, object]) -> None:
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
            waiting_data_count = sum(1 for entry in entries if entry["data_health"] == "waiting")
            data_status = "live" if live_data_count else ("waiting_for_data" if waiting_data_count else "no_data")
            cards.append(
                {
                    "name": organ_name,
                    "label": self.ORGAN_LABELS.get(organ_name, organ_name.title()),
                    "health": snapshot.get("health", "unknown"),
                    "data_health": self._data_health(data_status, str(snapshot.get("health", "unknown"))),
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
        for subfunction, key in (("capture", "capture_elapsed_ms"), ("asr", "asr_elapsed_ms")):
            elapsed_ms = details.get(key)
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
        state_age_s = detection_details.get("state_age_s", camera_details.get("state_age_s"))
        state_updated_at_ts = detection_details.get("state_updated_at_ts", camera_details.get("state_updated_at_ts"))
        frame_updated_at_ts = detection_details.get("frame_updated_at_ts", frame_captured_at_ts)
        backend = detection_details.get("backend", camera_details.get("backend", ""))
        service_status = detection_details.get("service_status", camera_details.get("service_status", detection_details.get("status")))
        state_path = detection_details.get("state_path", camera_details.get("state_path", ""))
        visual_tracking = body_snapshot.get("visual_tracking", {})
        if not isinstance(visual_tracking, dict):
            visual_tracking = {}
        registered_identity = body_snapshot.get("identity_registry", {})
        if not isinstance(registered_identity, dict):
            registered_identity = {}
        tracking_target = visual_tracking.get("target")
        if isinstance(tracking_target, dict):
            tracking_target = dict(tracking_target)
        else:
            tracking_target = None
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
        data_status = "sleeping" if service_status == "sleeping" else ("live" if frame_path else "waiting_for_frame")
        return {
            "enabled": bool(frame_path or detections or identity_candidates or registered_identity.get("registered")),
            "data_health": self._data_health(data_status, str(eye.get("health", "unknown"))),
            "data_status": data_status,
            "frame_available": bool(frame_path),
            "frame_url": "/vision/latest.jpg" if frame_path else None,
            "frame_captured_at_ts": frame_captured_at_ts,
            "camera_health": camera.get("health", "unknown"),
            "detection_health": detection.get("health", "unknown"),
            "identity_health": identity.get("health", "unknown"),
            "detection_status": detection_details.get("status", detection.get("health", "unknown")),
            "vision_service_status": service_status,
            "backend": backend,
            "state_path": state_path,
            "state_age_s": state_age_s,
            "state_updated_at_ts": state_updated_at_ts,
            "frame_updated_at_ts": frame_updated_at_ts,
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
            "top_detection_bbox": (detection_details.get("top_detection") or {}).get("bbox")
            if isinstance(detection_details.get("top_detection"), dict)
            else None,
            "frame_age_s": self._age_seconds(frame_captured_at_ts),
            "tracking_status": visual_tracking.get("status", "idle"),
            "tracking_source": visual_tracking.get("source", "inactive"),
            "tracking_updated_at_ts": visual_tracking.get("updated_at_ts"),
            "tracking_age_s": self._age_seconds(visual_tracking.get("updated_at_ts")),
            "tracking_running": bool(visual_tracking.get("running", False)),
            "tracking_target": tracking_target,
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
        if trace_details and not capture_details.get("dbfs"):
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
                    "voice_activity": trace_details.get("voice_activity", bool(trace_details.get("dbfs"))),
                    "streaming_vad": trace_details.get("streaming_vad"),
                    "vad_triggered": trace_details.get("vad_triggered"),
                    "vad_elapsed_ms": trace_details.get("vad_elapsed_ms"),
                    "captured_at_ts": trace_details.get("captured_at_ts") or trace_details.get("recorded_at_ts"),
                    "elapsed_ms": trace_details.get("capture_elapsed_ms"),
                }
            )
        if trace_details and not asr_details.get("transcript"):
            transcript_from_trace = str(trace_details.get("text", "") or "")
            asr_details.update(
                {
                    "driver": asr_details.get("driver", "recent_trace"),
                    "status": trace_details.get("asr_status") or ("transcribed" if transcript_from_trace else "no_transcript"),
                    "transcript": transcript_from_trace,
                    "voice_activity": trace_details.get("voice_activity", bool(trace_details.get("dbfs"))),
                    "asr_voice_activity": trace_details.get("asr_voice_activity"),
                    "min_asr_dbfs": trace_details.get("min_asr_dbfs"),
                    "recognizer_prewarmed": trace_details.get("recognizer_prewarmed"),
                    "recognizer_prewarm_error": trace_details.get("recognizer_prewarm_error"),
                    "speech_window_summary": trace_details.get("speech_window_summary"),
                    "captured_at_ts": trace_details.get("captured_at_ts") or trace_details.get("recorded_at_ts"),
                    "elapsed_ms": trace_details.get("asr_elapsed_ms"),
                }
            )
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
    def _age_seconds(timestamp: object) -> float | None:
        if not isinstance(timestamp, (int, float)):
            return None
        return round(max(0.0, time.time() - float(timestamp)), 2)


    @staticmethod
    def _build_memory_diagnostics(cognitive_snapshot: dict[str, object]) -> dict[str, object]:
        raw = cognitive_snapshot.get("memory_diagnostics", {})
        if not isinstance(raw, dict):
            raw = {}
        task_context = raw.get("task_context", {})
        if not isinstance(task_context, dict):
            task_context = {}
        last_recall = raw.get("last_recall", {})
        if not isinstance(last_recall, dict):
            last_recall = {}
        recall_filters = last_recall.get("recall_filters", {}) or task_context
        if not isinstance(recall_filters, dict):
            recall_filters = {}
        last_writeback = raw.get("last_writeback", {})
        if not isinstance(last_writeback, dict):
            last_writeback = {}
        return {
            "enabled": bool(raw or task_context or last_recall or last_writeback),
            "last_query": raw.get("last_query", ""),
            "task_type": task_context.get("task_type", ""),
            "recall_profile": task_context.get("recall_profile") or last_recall.get("recall_profile", ""),
            "allowed_sources": list(task_context.get("allowed_sources", []) or []),
            "blocked_sources": list(task_context.get("blocked_sources", []) or []),
            "allowed_memory_types": list(task_context.get("allowed_memory_types", []) or []),
            "preferred_modalities": list(task_context.get("preferred_modalities", []) or []),
            "organs": list(task_context.get("organs", []) or []),
            "selected_count": last_recall.get("selected_count", 0),
            "source_composition": last_recall.get("source_composition", {}),
            "selected_records": last_recall.get("selected_records", []),
            "recall_filters": dict(recall_filters),
            "last_writeback": dict(last_writeback),
        }

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
        return {
            "enabled": bool(loop.get("enabled")),
            "running": bool(loop.get("running")),
            "conversation_active": bool(loop.get("conversation_active")),
            "wake_word": loop.get("wake_word", ""),
            "sleep_word": loop.get("sleep_word", ""),
            "phase": loop.get("phase", "idle"),
            "phase_started_at_ts": loop.get("phase_started_at_ts"),
            "current_phase_elapsed_s": current_phase_elapsed_s,
            "last_status": loop.get("last_status", "idle"),
            "turn_count": loop.get("turn_count", 0),
            "last_transcript": loop.get("last_transcript", ""),
            "last_reply": loop.get("last_reply") or cognitive_snapshot.get("last_reply", ""),
            "last_error": loop.get("last_error", ""),
            "last_latency_s": loop.get("last_latency_s", {}),
            "last_completed_turn": loop.get("last_completed_turn", {}),
            "updated_at_ts": loop.get("updated_at_ts"),
            "learning_decision": cognitive_snapshot.get("learning_decision", ""),
            "last_review": cognitive_snapshot.get("last_review", {}),
            "last_llm_status": cognitive_snapshot.get("last_llm_status", {}),
        }

    @staticmethod
    def _build_neck_control_diagnostics(body_snapshot: dict[str, object]) -> dict[str, object]:
        raw = body_snapshot.get("neck_control", {})
        if not isinstance(raw, dict):
            raw = {}

        active_intent_raw = raw.get("active_intent", {})
        active_intent: object = active_intent_raw
        active_source = raw.get("active_source") or raw.get("source") or ""
        if isinstance(active_intent_raw, dict):
            active_intent = (
                active_intent_raw.get("target_name")
                or active_intent_raw.get("intent")
                or active_intent_raw.get("name")
                or active_intent_raw.get("id")
                or active_intent_raw.get("type")
                or ""
            )
            active_source = active_intent_raw.get("source") or active_source

        intents = raw.get("intents", [])
        intent_count = raw.get("intent_count")
        if not isinstance(intent_count, int):
            intent_count = len(intents) if isinstance(intents, list) else 0

        desired_angle = raw.get("desired_angle")
        if desired_angle is None:
            desired_angle = raw.get("target_angle")
        last_angle = raw.get("last_angle")
        if last_angle is None:
            last_angle = raw.get("current_angle")

        last_command_status = raw.get("last_command_status")
        if isinstance(last_command_status, dict):
            command_status_label = last_command_status.get("status", "")
        else:
            command_status_label = last_command_status or ""

        return {
            "enabled": bool(raw),
            "state": raw.get("state", "unavailable" if not raw else "unknown"),
            "active_intent": active_intent or "",
            "active_source": active_source or "",
            "desired_angle": desired_angle,
            "last_angle": last_angle,
            "suppressed_reason": raw.get("suppressed_reason", ""),
            "last_command_status": last_command_status or {},
            "last_command_status_label": command_status_label,
            "intent_count": intent_count,
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
        if data_status in {"live", "recent_trace", "played", "planned", "sleeping"}:
            return "healthy"
        if fallback_health == "unavailable":
            return "unavailable"
        if data_status in {"waiting_for_data", "waiting_for_frame", "waiting_for_action", "waiting_for_target"}:
            return "waiting"
        return "degraded"

    @staticmethod
    def _subfunction_data_status(
        *,
        organ_name: str,
        sub_name: str,
        status: str,
        details: dict[str, object],
    ) -> str:
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
            if status in {"listening", "listening_loop", "no_transcript", "short_transcript_ignored"}:
                return "live"
            if details.get("captured_at_ts") or details.get("payload_bytes") or details.get("transcript"):
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

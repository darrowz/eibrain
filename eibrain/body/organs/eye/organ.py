"""Eye organ implementation."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth
from eibrain.body.organs.base import BaseOrgan
from eibrain.body.runtime_linux import capture_frame, run_hailo_frame_inference


class EyeOrgan(BaseOrgan):
    name = "eye"
    subfunction_names = ("camera", "detection", "identity")

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)
        self._cache_ttl_s = self._read_float_config("detection", "refresh_interval_s", default=0.5)
        frame_dir = Path(tempfile.gettempdir()) / "eibrain-eye"
        frame_dir.mkdir(parents=True, exist_ok=True)
        self._frame_path = frame_dir / "latest.jpg"
        self._cached_heartbeat: OrganHealth | None = None
        self._cached_heartbeat_at = 0.0

    @property
    def latest_frame_path(self) -> str | None:
        if self._frame_path.exists():
            return str(self._frame_path)
        return None

    def passive_heartbeat(self) -> OrganHealth:
        if self._cached_heartbeat is not None:
            return self._cached_heartbeat
        subfunctions = {
            name: SubfunctionHealth(
                name=name,
                health="healthy",
                details={"driver": self._driver_kind(name), "status": "live_probe_skipped"},
            )
            for name in self.subfunction_names
        }
        return OrganHealth(organ=self.name, health="healthy", subfunctions=subfunctions)

    def heartbeat(self) -> OrganHealth:
        if not self._visual_runtime_enabled():
            return super().heartbeat()
        now_ts = time.time()
        if self._cached_heartbeat is not None and now_ts - self._cached_heartbeat_at < self._cache_ttl_s:
            return self._cached_heartbeat

        camera_state = self._camera_health(now_ts=now_ts)
        detection_state = self._detection_health(camera_state=camera_state, now_ts=now_ts)
        identity_state = self._identity_health(detection_state=detection_state, now_ts=now_ts)
        subfunctions = {
            "camera": camera_state,
            "detection": detection_state,
            "identity": identity_state,
        }
        statuses = [state.health for state in subfunctions.values()]
        if statuses and all(status == "healthy" for status in statuses):
            health = "healthy"
        elif any(status == "healthy" for status in statuses) or any(status == "degraded" for status in statuses):
            health = "degraded"
        else:
            health = "unavailable"
        self._cached_heartbeat = OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)
        self._cached_heartbeat_at = now_ts
        return self._cached_heartbeat

    def _visual_runtime_enabled(self) -> bool:
        return any(
            self.config.subfunctions.get(name) is not None
            and self.config.subfunctions[name].driver.kind != "noop"
            for name in ("camera", "detection", "identity")
        )

    def _camera_health(self, *, now_ts: float) -> SubfunctionHealth:
        if self._driver_kind("camera") == "noop":
            return self._subfunction_health("camera")
        config = self.config.subfunctions.get("camera")
        device = str(config.driver.extra.get("device", "/dev/video0")) if config is not None else "/dev/video0"
        started = time.perf_counter()
        probe = self.drivers["camera"].heartbeat()
        capture_result = capture_frame(device=device, output_path=self._frame_path)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        details = self._merge_probe_details(
            probe=probe.details,
            elapsed_ms=elapsed_ms,
            status="healthy" if capture_result.get("status") == "ok" else "capture_failed",
        )
        details.update(
            {
                "device": device,
                "frame_path": str(self._frame_path),
                "frame_captured_at_ts": now_ts,
                "capture_result": dict(capture_result.get("details", {})),
            }
        )
        if capture_result.get("status") == "ok":
            health = "healthy"
        else:
            health = "unavailable" if probe.status == "unavailable" else "degraded"
            capture_details = capture_result.get("details", {})
            if isinstance(capture_details, dict):
                details["error"] = capture_details.get("stderr") or capture_details.get("stdout") or "capture_failed"
        return SubfunctionHealth(name="camera", health=health, details=details)

    def _detection_health(
        self,
        *,
        camera_state: SubfunctionHealth,
        now_ts: float,
    ) -> SubfunctionHealth:
        if self._driver_kind("detection") == "noop":
            return self._subfunction_health("detection")
        config = self.config.subfunctions.get("detection")
        probe = self.drivers["detection"].heartbeat()
        probe_details = dict(probe.details)
        started = time.perf_counter()
        if camera_state.health != "healthy":
            details = self._merge_probe_details(
                probe=probe_details,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                status="camera_unavailable",
            )
            details.update(
                {
                    "frame_path": self.latest_frame_path,
                    "frame_captured_at_ts": now_ts,
                    "detections": [],
                    "detection_count": 0,
                    "scene_summary": "camera unavailable",
                    "error": "camera_unavailable",
                }
            )
            return SubfunctionHealth(name="detection", health="unavailable", details=details)

        hef_path = str(config.driver.extra.get("hef_path", "/usr/share/hailo-models/yolov5s_personface_h8l.hef")) if config is not None else "/usr/share/hailo-models/yolov5s_personface_h8l.hef"
        score_threshold = self._read_float_config("detection", "score_threshold", default=0.3)
        labels = self._read_label_config("detection", default=["person", "face"])
        inference = run_hailo_frame_inference(
            image_path=self._frame_path,
            hef_path=hef_path,
            labels=labels,
            score_threshold=score_threshold,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        inference_details = dict(inference.get("details", {}))
        detections = inference_details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        details = self._merge_probe_details(
            probe=probe_details,
            elapsed_ms=elapsed_ms,
            status="healthy" if inference.get("status") == "ok" else "inference_failed",
        )
        details.update(
            {
                "frame_path": str(self._frame_path),
                "frame_captured_at_ts": now_ts,
                "hef_path": hef_path,
                "score_threshold": score_threshold,
                "labels": labels,
                "detections": detections,
                "detection_count": len(detections),
                "top_detection": detections[0] if detections else None,
                "scene_labels": sorted({str(item.get("label", "unknown")) for item in detections}),
                "scene_summary": self._summarize_detections(detections),
                "inference_details": inference_details,
            }
        )
        if inference.get("status") == "ok":
            return SubfunctionHealth(name="detection", health="healthy", details=details)
        details["error"] = inference_details.get("error") or inference_details.get("stderr") or inference_details.get("reason")
        health = "unavailable" if probe.status == "unavailable" else "degraded"
        return SubfunctionHealth(name="detection", health=health, details=details)

    def _identity_health(
        self,
        *,
        detection_state: SubfunctionHealth,
        now_ts: float,
    ) -> SubfunctionHealth:
        if self._driver_kind("identity") == "noop":
            return self._subfunction_health("identity")
        probe = self.drivers["identity"].heartbeat()
        probe_details = dict(probe.details)
        started = time.perf_counter()
        detections = detection_state.details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        face_candidates = [item for item in detections if str(item.get("label")) == "face"]
        identity_candidates = [
            {
                "candidate_id": f"unknown-face-{index + 1}",
                "identity": "unknown",
                "score": candidate.get("score"),
                "bbox": candidate.get("bbox"),
            }
            for index, candidate in enumerate(face_candidates)
        ]
        if detection_state.health != "healthy":
            status = "detection_unavailable"
            health = "unavailable"
        elif identity_candidates:
            status = "observing_unknown_face"
            health = "healthy"
        else:
            status = "no_face_candidates"
            health = "healthy"
        details = self._merge_probe_details(
            probe=probe_details,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            status=status,
        )
        details.update(
            {
                "frame_path": self.latest_frame_path,
                "frame_captured_at_ts": now_ts,
                "identity_candidates": identity_candidates,
                "face_candidate_count": len(identity_candidates),
                "identity_summary": self._summarize_identity(identity_candidates, status=status),
            }
        )
        if health != "healthy":
            details["error"] = status
        return SubfunctionHealth(name="identity", health=health, details=details)

    def _driver_kind(self, name: str) -> str:
        config = self.config.subfunctions.get(name)
        if config is None:
            return "noop"
        return str(config.driver.kind)

    def _read_float_config(self, subfunction_name: str, key: str, *, default: float) -> float:
        config = self.config.subfunctions.get(subfunction_name)
        if config is None:
            return default
        value = config.driver.extra.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _read_label_config(self, subfunction_name: str, *, default: list[str]) -> list[str]:
        config = self.config.subfunctions.get(subfunction_name)
        if config is None:
            return list(default)
        raw = config.driver.extra.get("labels", default)
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return list(default)

    @staticmethod
    def _merge_probe_details(
        *,
        probe: dict[str, object],
        elapsed_ms: float,
        status: str,
    ) -> dict[str, object]:
        merged = dict(probe)
        merged["driver"] = merged.get("driver", "command")
        merged["elapsed_ms"] = elapsed_ms
        merged["status"] = status
        nested = merged.get("details", {})
        if not isinstance(nested, dict):
            nested = {}
        merged["details"] = nested
        return merged

    @staticmethod
    def _summarize_detections(detections: list[dict[str, object]]) -> str:
        if not detections:
            return "no detections in current frame"
        counts: dict[str, int] = {}
        for detection in detections:
            label = str(detection.get("label", "unknown"))
            counts[label] = counts.get(label, 0) + 1
        return ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))

    @staticmethod
    def _summarize_identity(identity_candidates: list[dict[str, object]], *, status: str) -> str:
        if identity_candidates:
            return f"{len(identity_candidates)} unknown face candidate(s)"
        if status == "detection_unavailable":
            return "identity chain blocked by detection"
        return "no recognizable face candidate in current frame"

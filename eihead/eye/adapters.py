"""Realtime hardware adapter scaffold for the eihead eye pipeline.

The classes here define a testable boundary for a future
``/dev/video0`` + ``/dev/hailo0`` GStreamer/Hailo implementation.  They do
not import GStreamer at module import time and they do not fall back to static
images as the primary path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
import time
from typing import Any

from .realtime import (
    REALTIME_STREAM_MODE,
    RealtimeDetection,
    RealtimeEyePipeline,
    RealtimeEyeStatus,
    RealtimeVisionFrame,
)


DeviceExists = Callable[[str], bool]
GstAvailable = Callable[[], bool]
FrameReader = Callable[[], RealtimeVisionFrame | Mapping[str, Any] | None]
DetectionReader = Callable[[RealtimeVisionFrame], Iterable[RealtimeDetection | Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class GStreamerHailoRealtimeConfig:
    """Configuration for the future realtime GStreamer/Hailo path."""

    camera_device: str = "/dev/video0"
    hailo_device: str = "/dev/hailo0"
    width: int = 640
    height: int = 480
    framerate: int = 30
    hef_path: str = ""
    model_id: str = "hailo"
    backend: str = "gstreamer_hailo"
    source_name: str = "gstreamer_hailo"
    mode: str = REALTIME_STREAM_MODE
    appsink_name: str = "eihead_realtime_sink"

    @property
    def device_paths(self) -> tuple[str, str]:
        return (self.camera_device, self.hailo_device)

    def pipeline_fields(self) -> dict[str, str]:
        """Return a deterministic pipeline description without importing Gst."""

        inference_parts = ["hailonet", f"device={self.hailo_device}"]
        if self.hef_path:
            inference_parts.append(f"hef-path={self.hef_path}")
        return {
            "mode": self.mode,
            "backend": self.backend,
            "camera_device": self.camera_device,
            "hailo_device": self.hailo_device,
            "source": f"v4l2src device={self.camera_device} do-timestamp=true",
            "caps": f"video/x-raw,width={int(self.width)},height={int(self.height)},framerate={int(self.framerate)}/1",
            "convert": "videoconvert",
            "inference": " ".join(inference_parts),
            "sink": (
                f"appsink name={self.appsink_name} emit-signals=true "
                "sync=false max-buffers=1 drop=true"
            ),
        }

    def build_pipeline_description(self) -> str:
        fields = self.pipeline_fields()
        return " ! ".join(
            [
                fields["source"],
                fields["caps"],
                fields["convert"],
                fields["inference"],
                fields["sink"],
            ]
        )


@dataclass(frozen=True, slots=True)
class AdapterReadiness:
    ready: bool
    status: str
    message: str
    missing_devices: tuple[str, ...] = ()


class GStreamerHailoFrameSource:
    """Realtime frame source boundary for the future GStreamer appsink."""

    mode = REALTIME_STREAM_MODE

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig | None = None,
        *,
        device_exists: DeviceExists | None = None,
        gst_available: GstAvailable | None = None,
        frame_reader: FrameReader | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config or GStreamerHailoRealtimeConfig()
        self.source_name = self.config.source_name
        self.backend = self.config.backend
        self._device_exists = device_exists or _default_device_exists
        self._gst_available = gst_available or _default_gst_available
        self._frame_reader = frame_reader
        self._clock = clock

    @property
    def placeholder(self) -> bool:
        return not self.readiness().ready

    @property
    def not_wired(self) -> bool:
        return not self.readiness().ready

    def readiness(self) -> AdapterReadiness:
        readiness = _readiness(self.config, device_exists=self._device_exists, gst_available=self._gst_available)
        if not readiness.ready:
            return readiness
        if self._frame_reader is None:
            return AdapterReadiness(
                ready=False,
                status="not_wired",
                message="realtime frame reader is not wired",
            )
        return readiness

    def pipeline_fields(self) -> dict[str, str]:
        return self.config.pipeline_fields()

    def build_pipeline_description(self) -> str:
        return self.config.build_pipeline_description()

    def next_frame(self) -> RealtimeVisionFrame | None:
        if not self.readiness().ready or self._frame_reader is None:
            return None
        try:
            raw_frame = self._frame_reader()
        except Exception as exc:
            raise AdapterRuntimeError(
                f"realtime frame reader failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        if raw_frame is None:
            return None
        return _coerce_realtime_frame(raw_frame, config=self.config, clock=self._clock)


class GStreamerHailoDetector:
    """Detector boundary that normalizes future Hailo results."""

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig | None = None,
        *,
        device_exists: DeviceExists | None = None,
        gst_available: GstAvailable | None = None,
        detection_reader: DetectionReader | None = None,
    ) -> None:
        self.config = config or GStreamerHailoRealtimeConfig()
        self.backend = self.config.backend
        self._device_exists = device_exists or _default_device_exists
        self._gst_available = gst_available or _default_gst_available
        self._detection_reader = detection_reader

    @property
    def placeholder(self) -> bool:
        return not self.readiness().ready

    @property
    def not_wired(self) -> bool:
        return not self.readiness().ready

    def readiness(self) -> AdapterReadiness:
        readiness = _readiness(self.config, device_exists=self._device_exists, gst_available=self._gst_available)
        if not readiness.ready:
            return readiness
        if self._detection_reader is None:
            return AdapterReadiness(
                ready=False,
                status="not_wired",
                message="realtime detection reader is not wired",
            )
        return readiness

    def detect(self, frame: RealtimeVisionFrame) -> list[RealtimeDetection]:
        if not self.readiness().ready or self._detection_reader is None:
            return []
        try:
            raw_detections = list(self._detection_reader(frame))
        except Exception as exc:
            raise AdapterRuntimeError(
                f"realtime detection reader failed: {exc.__class__.__name__}: {exc}"
            ) from exc
        return [
            normalize_hailo_detection(raw_detection, frame=frame, config=self.config)
            for raw_detection in raw_detections
        ]


class GStreamerHailoRealtimeAdapter:
    """Small composition wrapper for source, detector, and realtime status."""

    def __init__(
        self,
        config: GStreamerHailoRealtimeConfig | None = None,
        *,
        device_exists: DeviceExists | None = None,
        gst_available: GstAvailable | None = None,
        frame_reader: FrameReader | None = None,
        detection_reader: DetectionReader | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config or GStreamerHailoRealtimeConfig()
        self._device_exists = device_exists or _default_device_exists
        self._gst_available = gst_available or _default_gst_available
        self._frame_reader = frame_reader
        self._detection_reader = detection_reader
        self.frame_source = GStreamerHailoFrameSource(
            self.config,
            device_exists=self._device_exists,
            gst_available=self._gst_available,
            frame_reader=frame_reader,
            clock=clock,
        )
        self.detector = GStreamerHailoDetector(
            self.config,
            device_exists=self._device_exists,
            gst_available=self._gst_available,
            detection_reader=detection_reader,
        )
        self._pipeline = RealtimeEyePipeline(
            frame_source=self.frame_source,
            detector=self.detector,
            backend=self.config.backend,
            mode=self.config.mode,
            clock=clock,
        )
        self._last_status = self._initial_status()

    def readiness(self) -> AdapterReadiness:
        readiness = _readiness(self.config, device_exists=self._device_exists, gst_available=self._gst_available)
        if not readiness.ready:
            return readiness
        if self._frame_reader is None:
            return AdapterReadiness(
                ready=False,
                status="not_wired",
                message="realtime frame reader is not wired",
            )
        if self._detection_reader is None:
            return AdapterReadiness(
                ready=False,
                status="not_wired",
                message="realtime detection reader is not wired",
            )
        return readiness

    def pipeline_fields(self) -> dict[str, str]:
        return self.config.pipeline_fields()

    def build_pipeline_description(self) -> str:
        return self.config.build_pipeline_description()

    def status(self) -> RealtimeEyeStatus:
        readiness = self.readiness()
        if not readiness.ready:
            self._last_status = _not_wired_status(self.config, readiness.message)
        return self._last_status

    def poll(self) -> RealtimeEyeStatus:
        readiness = self.readiness()
        if not readiness.ready:
            self._last_status = _not_wired_status(self.config, readiness.message)
            return self._last_status

        try:
            payload = self._pipeline.process_next()
        except AdapterRuntimeError as exc:
            self._last_status = _not_wired_status(self.config, str(exc))
            return self._last_status
        except Exception as exc:  # pragma: no cover - defensive guard for hardware backends.
            self._last_status = _not_wired_status(
                self.config,
                f"realtime adapter poll failed: {exc.__class__.__name__}: {exc}",
            )
            return self._last_status
        self._last_status = _status_from_payload(payload)
        return self._last_status

    def _initial_status(self) -> RealtimeEyeStatus:
        readiness = self.readiness()
        if not readiness.ready:
            return _not_wired_status(self.config, readiness.message)
        return RealtimeEyeStatus(
            mode=self.config.mode,
            status="waiting_for_frame",
            backend=self.config.backend,
            source="eihead.eye.adapters",
            placeholder=False,
            not_wired=False,
            message="realtime adapter is wired; waiting for frames",
        )


def normalize_hailo_detection(
    raw_detection: RealtimeDetection | Mapping[str, Any],
    *,
    frame: RealtimeVisionFrame,
    config: GStreamerHailoRealtimeConfig | None = None,
) -> RealtimeDetection:
    """Convert a raw Hailo/GStreamer detection payload to RealtimeDetection."""

    if isinstance(raw_detection, RealtimeDetection):
        return raw_detection

    adapter_config = config or GStreamerHailoRealtimeConfig()
    bbox = _normalize_bbox(_raw_bbox(raw_detection), frame=frame)
    class_id = _safe_int(raw_detection.get("class_id"))
    label = _first_text(raw_detection, ("label", "class_name", "name")) or (
        f"class_{class_id}" if class_id is not None else "unknown"
    )
    confidence = _safe_float(
        _first_value(raw_detection, ("confidence", "score", "probability")),
        default=0.0,
    )
    attributes = raw_detection.get("attributes", {})
    return RealtimeDetection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        class_id=class_id,
        track_id=raw_detection.get("track_id"),
        source=adapter_config.source_name,
        model_id=adapter_config.model_id,
        attributes=attributes if isinstance(attributes, Mapping) else {},
    )


def _readiness(
    config: GStreamerHailoRealtimeConfig,
    *,
    device_exists: DeviceExists,
    gst_available: GstAvailable,
) -> AdapterReadiness:
    missing_devices = tuple(path for path in config.device_paths if not device_exists(path))
    if missing_devices:
        return AdapterReadiness(
            ready=False,
            status="not_wired",
            missing_devices=missing_devices,
            message=f"missing realtime devices: {', '.join(missing_devices)}",
        )
    if not gst_available():
        return AdapterReadiness(
            ready=False,
            status="not_wired",
            message="GStreamer backend is not installed",
        )
    return AdapterReadiness(ready=True, status="ready", message="realtime adapter is wired")


def _not_wired_status(config: GStreamerHailoRealtimeConfig, message: str) -> RealtimeEyeStatus:
    return RealtimeEyeStatus(
        mode=config.mode,
        status="not_wired",
        backend=config.backend,
        source="eihead.eye.adapters",
        placeholder=True,
        not_wired=True,
        message=message,
    )


def _coerce_realtime_frame(
    raw_frame: RealtimeVisionFrame | Mapping[str, Any],
    *,
    config: GStreamerHailoRealtimeConfig,
    clock: Callable[[], float],
) -> RealtimeVisionFrame:
    if isinstance(raw_frame, RealtimeVisionFrame):
        if raw_frame.is_compat_static:
            raise ValueError("GStreamerHailoFrameSource only accepts realtime frames")
        return raw_frame

    metadata = raw_frame.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    raw_timestamp = raw_frame.get("timestamp", raw_frame.get("captured_at_ts"))
    if raw_timestamp is None:
        timestamp = float(clock())
    else:
        try:
            timestamp = float(raw_timestamp)
        except (TypeError, ValueError):
            timestamp = float(clock())
    return RealtimeVisionFrame(
        frame_id=str(raw_frame.get("frame_id") or f"gstreamer-hailo-{int(clock() * 1000)}"),
        timestamp=timestamp,
        width=_safe_int(raw_frame.get("width")) or config.width,
        height=_safe_int(raw_frame.get("height")) or config.height,
        source=config.source_name,
        mode=config.mode,
        payload=raw_frame.get("payload"),
        metadata={
            **dict(metadata),
            "backend": config.backend,
            "camera_device": config.camera_device,
            "hailo_device": config.hailo_device,
        },
    )


def _status_from_payload(payload: Mapping[str, Any]) -> RealtimeEyeStatus:
    detections = [_detection_from_payload(item) for item in payload.get("detections", [])]
    top_detection_payload = payload.get("top_detection")
    top_detection = (
        _detection_from_payload(top_detection_payload)
        if isinstance(top_detection_payload, Mapping)
        else (detections[0] if detections else None)
    )
    return RealtimeEyeStatus(
        schema=str(payload.get("schema", "eihead.eye.realtime_status.v1")),
        mode=str(payload.get("mode", REALTIME_STREAM_MODE)),
        status=str(payload.get("status", "waiting_for_frame")),
        backend=str(payload.get("backend", "gstreamer_hailo")),
        frame_count=int(payload.get("frame_count", 0) or 0),
        detection_count=int(payload.get("detection_count", 0) or 0),
        fps=float(payload.get("fps", 0.0) or 0.0),
        last_frame_id=str(payload.get("last_frame_id", "") or ""),
        last_frame_age=payload.get("last_frame_age"),
        last_frame_captured_at_ts=payload.get("last_frame_captured_at_ts"),
        top_detection=top_detection,
        detections=detections,
        source=str(payload.get("source", "eihead.eye.adapters")),
        placeholder=bool(payload.get("placeholder", False)),
        not_wired=bool(payload.get("not_wired", False)),
        compatibility_mode=bool(payload.get("compatibility_mode", False)),
        message=str(payload.get("message", "") or ""),
    )


def _detection_from_payload(payload: Any) -> RealtimeDetection:
    if isinstance(payload, RealtimeDetection):
        return payload
    if not isinstance(payload, Mapping):
        return RealtimeDetection(label="unknown", confidence=0.0, bbox=None)
    return RealtimeDetection(
        label=str(payload.get("label", "unknown")),
        confidence=_safe_float(payload.get("confidence", payload.get("score")), default=0.0),
        bbox=payload.get("bbox") if isinstance(payload.get("bbox"), Mapping) else None,
        class_id=_safe_int(payload.get("class_id")),
        track_id=payload.get("track_id"),
        source=str(payload.get("source", "gstreamer_hailo")),
        model_id=str(payload.get("model_id", "")),
        attributes=payload.get("attributes") if isinstance(payload.get("attributes"), Mapping) else {},
    )


def _raw_bbox(raw_detection: Mapping[str, Any]) -> Any:
    return _first_value(raw_detection, ("bbox", "box", "bounds"))


def _normalize_bbox(raw_bbox: Any, *, frame: RealtimeVisionFrame) -> dict[str, float] | None:
    if raw_bbox is None:
        return None

    if isinstance(raw_bbox, Mapping):
        x_min = _first_value(raw_bbox, ("x_min", "xmin", "x1", "left"))
        y_min = _first_value(raw_bbox, ("y_min", "ymin", "y1", "top"))
        x_max = _first_value(raw_bbox, ("x_max", "xmax", "x2", "right"))
        y_max = _first_value(raw_bbox, ("y_max", "ymax", "y2", "bottom"))
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        x_min, y_min, x_max, y_max = raw_bbox
    else:
        return None

    bbox = {
        "x_min": _safe_float(x_min, default=0.0),
        "y_min": _safe_float(y_min, default=0.0),
        "x_max": _safe_float(x_max, default=0.0),
        "y_max": _safe_float(y_max, default=0.0),
    }
    width = _safe_float(frame.width, default=0.0)
    height = _safe_float(frame.height, default=0.0)
    if width > 0 and max(abs(bbox["x_min"]), abs(bbox["x_max"])) > 1.0:
        bbox["x_min"] = bbox["x_min"] / width
        bbox["x_max"] = bbox["x_max"] / width
    if height > 0 and max(abs(bbox["y_min"]), abs(bbox["y_max"])) > 1.0:
        bbox["y_min"] = bbox["y_min"] / height
        bbox["y_max"] = bbox["y_max"] / height
    return bbox


def _first_text(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    value = _first_value(mapping, keys)
    return str(value) if value is not None else ""


def _first_value(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_device_exists(path: str) -> bool:
    return Path(path).exists()


def _default_gst_available() -> bool:
    if importlib.util.find_spec("gi") is None:
        return False
    try:
        gi = importlib.import_module("gi")
        gi.require_version("Gst", "1.0")
        Gst = importlib.import_module("gi.repository.Gst")
        Gst.init(None)
        element_factory = getattr(Gst, "ElementFactory", None)
        if element_factory is None:
            return False
        return all(
            element_factory.find(element_name) is not None
            for element_name in ("v4l2src", "videoconvert", "appsink", "hailonet")
        )
    except Exception:
        return False


class AdapterRuntimeError(RuntimeError):
    """Raised when a realtime adapter callback fails during polling."""


__all__ = [
    "AdapterRuntimeError",
    "AdapterReadiness",
    "GStreamerHailoDetector",
    "GStreamerHailoFrameSource",
    "GStreamerHailoRealtimeAdapter",
    "GStreamerHailoRealtimeConfig",
    "normalize_hailo_detection",
    "RealtimeDetection",
    "RealtimeEyeStatus",
    "RealtimeVisionFrame",
]

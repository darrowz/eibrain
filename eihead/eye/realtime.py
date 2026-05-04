"""Realtime eye pipeline contracts for honjia.

This module intentionally does not open ``/dev/video0`` or ``/dev/hailo0``.
It defines the realtime-first boundary that concrete camera/Hailo adapters
will plug into. Static frame inputs are kept only as an explicit compatibility
source for tests and migration fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol


REALTIME_STREAM_MODE = "realtime_stream"
COMPAT_STATIC_FRAME_MODE = "compat_static_frame"


class FrameSource(Protocol):
    mode: str

    def next_frame(self) -> "RealtimeVisionFrame | None":
        """Return the next frame, or ``None`` when no frame is currently available."""


class FrameDetector(Protocol):
    backend: str

    def detect(self, frame: "RealtimeVisionFrame") -> Iterable["RealtimeDetection | Mapping[str, Any]"]:
        """Run detection for one frame and return normalized or raw detections."""


@dataclass(frozen=True, slots=True, init=False)
class RealtimeVisionFrame:
    frame_id: str
    timestamp: float
    width: int | None = None
    height: int | None = None
    source: str = "camera"
    mode: str = REALTIME_STREAM_MODE
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        frame_id: str,
        timestamp: float | None = None,
        captured_at_ts: float | None = None,
        width: int | None = None,
        height: int | None = None,
        source: str = "camera",
        mode: str = REALTIME_STREAM_MODE,
        payload: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        frame_ts = timestamp if timestamp is not None else captured_at_ts
        if frame_ts is None:
            raise TypeError("RealtimeVisionFrame requires timestamp")
        object.__setattr__(self, "frame_id", str(frame_id))
        object.__setattr__(self, "timestamp", float(frame_ts))
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        source_value = str(source)
        mode_value = str(mode)
        if mode_value == COMPAT_STATIC_FRAME_MODE and source_value != COMPAT_STATIC_FRAME_MODE:
            raise ValueError("compat static frames must use compat_static_frame source")
        object.__setattr__(self, "source", source_value)
        object.__setattr__(self, "mode", mode_value)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def captured_at_ts(self) -> float:
        return self.timestamp

    @property
    def is_realtime(self) -> bool:
        return self.mode == REALTIME_STREAM_MODE

    @property
    def is_compat_static(self) -> bool:
        return self.source == COMPAT_STATIC_FRAME_MODE or self.mode == COMPAT_STATIC_FRAME_MODE

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "captured_at_ts": self.captured_at_ts,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "mode": self.mode,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True, init=False)
class RealtimeDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float] | dict[str, float] | None
    class_id: int | None = None
    track_id: str | int | None = None
    source: str = "detector"
    model_id: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        label: str,
        confidence: float | None = None,
        score: float | None = None,
        bbox: tuple[float, float, float, float] | Mapping[str, float] | None = None,
        class_id: int | None = None,
        track_id: str | int | None = None,
        source: str = "detector",
        model_id: str = "",
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        detection_confidence = confidence if confidence is not None else score
        object.__setattr__(self, "label", str(label))
        object.__setattr__(self, "confidence", float(detection_confidence or 0.0))
        object.__setattr__(self, "bbox", _normalize_bbox(bbox))
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "source", str(source))
        object.__setattr__(self, "model_id", str(model_id))
        object.__setattr__(self, "attributes", dict(attributes or {}))

    @property
    def score(self) -> float:
        return self.confidence

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "confidence": round(float(self.confidence), 6),
            "score": round(float(self.score), 6),
            "bbox": _bbox_to_dict(self.bbox),
            "source": self.source,
        }
        if self.class_id is not None:
            normalized_class_id = _safe_int(self.class_id)
            if normalized_class_id is not None:
                payload["class_id"] = normalized_class_id
        if self.track_id is not None:
            payload["track_id"] = self.track_id
        if self.model_id:
            payload["model_id"] = self.model_id
        if self.attributes:
            payload["attributes"] = dict(self.attributes)
        return payload


@dataclass(frozen=True, slots=True)
class RealtimeEyeStatus:
    schema: str = "eihead.eye.realtime_status.v1"
    mode: str = REALTIME_STREAM_MODE
    status: str = "not_wired"
    backend: str = "placeholder"
    frame_count: int = 0
    detection_count: int = 0
    fps: float = 0.0
    last_frame_id: str = ""
    last_frame_age: float | None = None
    last_frame_captured_at_ts: float | None = None
    top_detection: RealtimeDetection | None = None
    detections: list[RealtimeDetection] = field(default_factory=list)
    source: str = "eihead.eye"
    placeholder: bool = True
    not_wired: bool = True
    compatibility_mode: bool = False
    message: str = "realtime eye pipeline is not wired"

    @property
    def last_frame_age_s(self) -> float | None:
        return self.last_frame_age

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mode": self.mode,
            "status": self.status,
            "backend": self.backend,
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
            "fps": self.fps,
            "last_frame_id": self.last_frame_id,
            "last_frame_age": self.last_frame_age,
            "last_frame_age_s": self.last_frame_age_s,
            "last_frame_captured_at_ts": self.last_frame_captured_at_ts,
            "top_detection": self.top_detection.to_dict() if self.top_detection else None,
            "detections": [item.to_dict() for item in self.detections],
            "source": self.source,
            "placeholder": self.placeholder,
            "not_wired": self.not_wired,
            "compatibility_mode": self.compatibility_mode,
            "message": self.message,
        }


class CompatStaticFrameSource:
    """Explicit migration-only source for single-image tests/fallbacks."""

    mode = COMPAT_STATIC_FRAME_MODE
    source_name = COMPAT_STATIC_FRAME_MODE

    def __init__(
        self,
        *,
        frame_id: str = "compat-static-frame",
        frame_path: str = "",
        width: int | None = None,
        height: int | None = None,
        captured_at_ts: float | None = None,
    ) -> None:
        self.frame_id = frame_id
        self.frame_path = frame_path
        self.width = width
        self.height = height
        self.captured_at_ts = captured_at_ts
        self._used = False

    def next_frame(self) -> RealtimeVisionFrame | None:
        if self._used:
            return None
        self._used = True
        return RealtimeVisionFrame(
            frame_id=self.frame_id,
            timestamp=self.captured_at_ts if self.captured_at_ts is not None else time.time(),
            width=self.width,
            height=self.height,
            source="compat_static_frame",
            mode=COMPAT_STATIC_FRAME_MODE,
            metadata={"frame_path": self.frame_path} if self.frame_path else {},
        )


class RealtimeEyePipeline:
    """Small realtime-first pipeline with injectable frame source and detector."""

    def __init__(
        self,
        *,
        frame_source: FrameSource | None = None,
        detector: FrameDetector | None = None,
        backend: str = "not_wired",
        mode: str = REALTIME_STREAM_MODE,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.frame_source = frame_source
        self.detector = detector
        self.backend = getattr(detector, "backend", None) or backend or "not_wired"
        self.mode = mode
        self.clock = clock
        self._frame_count = 0
        self._detection_count = 0
        self._first_frame_ts: float | None = None
        self._last_frame: RealtimeVisionFrame | None = None
        self._last_detections: list[RealtimeDetection] = []
        self._frame_iterator: Iterator[RealtimeVisionFrame] | None = None
        self._last_status = RealtimeEyeStatus(mode=mode, backend=self.backend)

    @classmethod
    def placeholder(cls, *, backend: str = "not_wired") -> "RealtimeEyePipeline":
        return cls(backend=backend)

    def status(self) -> dict[str, Any]:
        return self._last_status.to_dict()

    def run(self, *, max_frames: int | None = None) -> RealtimeEyeStatus:
        started_at_ts = float(self.clock())
        if self.frame_source is None or self.detector is None:
            self._last_status = RealtimeEyeStatus(
                mode=self.mode,
                status="not_wired",
                backend=self.backend,
                source="eihead.eye.realtime",
                placeholder=True,
                not_wired=True,
                message="realtime frame source or detector is not wired",
            )
            return self._last_status

        processed_at_ts = started_at_ts
        processed_this_run = 0
        for frame in self._iter_frames():
            processed_at_ts = float(self.clock())
            self._process_frame(frame)
            processed_this_run += 1
            if max_frames is not None and processed_this_run >= max_frames:
                break

        observed_at_ts = float(self.clock())
        self._last_status = self._build_status(
            now_ts=observed_at_ts,
            fps_ts=processed_at_ts,
            started_at_ts=started_at_ts,
            status="ok" if self._last_frame is not None else "waiting_for_frame",
        )
        return self._last_status

    def process_next(self) -> dict[str, Any]:
        now_ts = float(self.clock())
        if self.frame_source is None or self.detector is None:
            self._last_status = RealtimeEyeStatus(
                mode=self.mode,
                status="not_wired",
                backend=self.backend,
                source="eihead.eye.realtime",
                placeholder=True,
                not_wired=True,
                message="realtime frame source or detector is not wired",
            )
            return self.status()

        frame = self._next_frame()
        if frame is None:
            self._last_status = RealtimeEyeStatus(
                mode=self._effective_mode(),
                status="waiting_for_frame",
                backend=self._effective_backend(),
                frame_count=self._frame_count,
                detection_count=self._detection_count,
                fps=self._fps(now_ts, self._first_frame_ts or now_ts),
                source="eihead.eye.realtime",
                placeholder=self._placeholder(),
                not_wired=self._not_wired(),
                compatibility_mode=self._compatibility_mode(),
                message="no realtime frame available",
            )
            return self.status()

        self._process_frame(frame)
        self._last_status = self._build_status(now_ts=now_ts, fps_ts=now_ts, started_at_ts=self._first_frame_ts or now_ts)
        return self.status()

    def _iter_frames(self) -> Iterator[RealtimeVisionFrame]:
        while True:
            frame = self._next_frame()
            if frame is None:
                return
            yield frame

    def _next_frame(self) -> RealtimeVisionFrame | None:
        if self.frame_source is None:
            return None
        if hasattr(self.frame_source, "frames"):
            if self._frame_iterator is None:
                self._frame_iterator = iter(self.frame_source.frames())  # type: ignore[attr-defined]
            return next(self._frame_iterator, None)
        return self.frame_source.next_frame()

    def _process_frame(self, frame: RealtimeVisionFrame) -> None:
        if self.detector is None:
            return
        self._frame_count += 1
        self._first_frame_ts = self._first_frame_ts or frame.timestamp
        detections = [_coerce_detection(item) for item in self.detector.detect(frame)]
        detections.sort(key=lambda item: item.confidence, reverse=True)
        self._detection_count += len(detections)
        self._last_frame = frame
        self._last_detections = detections

    def _build_status(
        self,
        *,
        now_ts: float,
        fps_ts: float,
        started_at_ts: float,
        status: str = "ok",
    ) -> RealtimeEyeStatus:
        frame = self._last_frame
        compatibility_mode = bool(frame and frame.is_compat_static)
        effective_status = "compat_static" if compatibility_mode and status == "ok" else status
        return RealtimeEyeStatus(
            mode=frame.mode if frame else self._effective_mode(),
            status=effective_status,
            backend=self._effective_backend(),
            frame_count=self._frame_count,
            detection_count=self._detection_count,
            fps=self._fps(fps_ts, started_at_ts),
            last_frame_id=frame.frame_id if frame else "",
            last_frame_age=round(max(0.0, now_ts - frame.timestamp), 4) if frame else None,
            last_frame_captured_at_ts=frame.timestamp if frame else None,
            top_detection=self._last_detections[0] if self._last_detections else None,
            detections=list(self._last_detections),
            source="eihead.eye.realtime",
            placeholder=self._placeholder(),
            not_wired=self._not_wired(),
            compatibility_mode=compatibility_mode,
            message=(
                "compat static frame processed; realtime stream remains primary"
                if compatibility_mode
                else "realtime frame processed"
            ),
        )

    def _effective_mode(self) -> str:
        return getattr(self.frame_source, "mode", self.mode)

    def _effective_backend(self) -> str:
        return getattr(self.detector, "backend", self.backend)

    def _compatibility_mode(self) -> bool:
        source_name = getattr(self.frame_source, "source_name", "")
        return self._effective_mode() == COMPAT_STATIC_FRAME_MODE or source_name == COMPAT_STATIC_FRAME_MODE

    def _placeholder(self) -> bool:
        return bool(getattr(self.detector, "placeholder", False))

    def _not_wired(self) -> bool:
        return bool(getattr(self.detector, "not_wired", self._placeholder()))

    def _fps(self, now_ts: float, started_at_ts: float) -> float:
        if self._frame_count <= 0:
            return 0.0
        elapsed_s = max(0.001, now_ts - started_at_ts)
        return round(self._frame_count / elapsed_s, 3)


def _coerce_detection(raw: RealtimeDetection | Mapping[str, Any]) -> RealtimeDetection:
    if isinstance(raw, RealtimeDetection):
        return raw
    bbox = raw.get("bbox", {}) if isinstance(raw, Mapping) else {}
    if not isinstance(bbox, Mapping):
        bbox = {}
    return RealtimeDetection(
        label=str(raw.get("label", "unknown") if isinstance(raw, Mapping) else "unknown"),
        confidence=float(raw.get("confidence", raw.get("score", 0.0)) if isinstance(raw, Mapping) else 0.0),
        bbox=bbox,
        class_id=raw.get("class_id") if isinstance(raw, Mapping) and raw.get("class_id") is not None else None,
        track_id=raw.get("track_id") if isinstance(raw, Mapping) else None,
        source=str(raw.get("source", "detector") if isinstance(raw, Mapping) else "detector"),
        model_id=str(raw.get("model_id", "") if isinstance(raw, Mapping) else ""),
        attributes=dict(raw.get("attributes", {}) if isinstance(raw, Mapping) and isinstance(raw.get("attributes"), Mapping) else {}),
    )


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_bbox(
    bbox: tuple[float, float, float, float] | Mapping[str, float] | None,
) -> tuple[float, float, float, float] | dict[str, float] | None:
    if bbox is None:
        return None
    if isinstance(bbox, Mapping):
        return {str(key): float(value) for key, value in bbox.items()}
    return tuple(float(value) for value in bbox)  # type: ignore[return-value]


def _bbox_to_dict(bbox: tuple[float, float, float, float] | dict[str, float] | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return {key: round(float(value), 6) for key, value in bbox.items()}
    x_min, y_min, x_max, y_max = bbox
    return {
        "x_min": round(float(x_min), 6),
        "y_min": round(float(y_min), 6),
        "x_max": round(float(x_max), 6),
        "y_max": round(float(y_max), 6),
    }


__all__ = [
    "COMPAT_STATIC_FRAME_MODE",
    "REALTIME_STREAM_MODE",
    "CompatStaticFrameSource",
    "FrameDetector",
    "FrameSource",
    "RealtimeDetection",
    "RealtimeEyePipeline",
    "RealtimeEyeStatus",
    "RealtimeVisionFrame",
]

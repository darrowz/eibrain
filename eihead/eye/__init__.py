"""eihead-native realtime eye primitives."""

from .adapters import (
    AdapterRuntimeError,
    AdapterReadiness,
    GStreamerHailoDetector,
    GStreamerHailoFrameSource,
    GStreamerHailoRealtimeAdapter,
    GStreamerHailoRealtimeConfig,
    normalize_hailo_detection,
)
from .realtime import (
    CompatStaticFrameSource,
    RealtimeDetection,
    RealtimeEyePipeline,
    RealtimeEyeStatus,
    RealtimeVisionFrame,
)

__all__ = [
    "AdapterReadiness",
    "AdapterRuntimeError",
    "CompatStaticFrameSource",
    "GStreamerHailoDetector",
    "GStreamerHailoFrameSource",
    "GStreamerHailoRealtimeAdapter",
    "GStreamerHailoRealtimeConfig",
    "RealtimeDetection",
    "RealtimeEyePipeline",
    "RealtimeEyeStatus",
    "RealtimeVisionFrame",
    "normalize_hailo_detection",
]

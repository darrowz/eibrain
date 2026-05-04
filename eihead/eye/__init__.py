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
from .gstreamer import GStreamerAppSinkFrameReader
from .hailo_metadata import HailoMetadataParseError, parse_hailo_detections
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
    "GStreamerAppSinkFrameReader",
    "HailoMetadataParseError",
    "RealtimeDetection",
    "RealtimeEyePipeline",
    "RealtimeEyeStatus",
    "RealtimeVisionFrame",
    "normalize_hailo_detection",
    "parse_hailo_detections",
]

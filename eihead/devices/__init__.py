"""Device adapters for eihead-native modules."""

from .audio import (
    AcousticFrontendReadiness,
    AudioDeviceCandidate,
    PlaybackInterruptionPlan,
    build_aplay_command,
    build_arecord_command,
    evaluate_audio_frontend_readiness,
    select_preferred_input,
)
from .neck_servo import NeckServoCommandAdapter

__all__ = [
    "AcousticFrontendReadiness",
    "AudioDeviceCandidate",
    "NeckServoCommandAdapter",
    "PlaybackInterruptionPlan",
    "build_aplay_command",
    "build_arecord_command",
    "evaluate_audio_frontend_readiness",
    "select_preferred_input",
]

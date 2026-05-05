from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Sequence


AudioDeviceKind = Literal["input", "output", "loopback"]


@dataclass(frozen=True)
class AudioDeviceCandidate:
    name: str
    kind: AudioDeviceKind
    device: str
    score: int
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def select_preferred_input(
    candidates: Iterable[AudioDeviceCandidate],
    preferred_keywords: Sequence[str] = ("U4K",),
) -> AudioDeviceCandidate:
    input_candidates = [candidate for candidate in candidates if candidate.kind == "input"]
    if not input_candidates:
        raise ValueError("no input audio candidates available")

    ranked = [_rank_input_candidate(candidate, preferred_keywords) for candidate in input_candidates]
    return max(ranked, key=lambda candidate: candidate.score)


def _rank_input_candidate(
    candidate: AudioDeviceCandidate,
    preferred_keywords: Sequence[str],
) -> AudioDeviceCandidate:
    name_upper = candidate.name.upper()
    score = candidate.score
    reason_parts = [candidate.reason]
    metadata = dict(candidate.metadata)

    for keyword in preferred_keywords:
        if keyword.upper() in name_upper:
            score += 10_000
            reason_parts.append(f"preferred field microphone keyword matched: {keyword}")
            metadata["preferred_keyword"] = keyword
            break

    if "SPA3700" in name_upper:
        score -= 1_000
        reason_parts.append("SPA3700 input not confirmed usable")
        metadata["degraded"] = True
        metadata["deprioritized"] = "SPA3700 input not confirmed usable"

    return replace(
        candidate,
        score=score,
        reason="; ".join(part for part in reason_parts if part),
        metadata=metadata,
    )


def build_arecord_command(
    device: str,
    sample_rate: int = 16_000,
    channels: int = 1,
    frame_ms: int = 60,
) -> list[str]:
    return [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "--period-time",
        str(frame_ms * 1_000),
    ]


def build_aplay_command(
    device: str,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> list[str]:
    return [
        "aplay",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
    ]


@dataclass(frozen=True)
class AcousticFrontendReadiness:
    aec: bool
    ns: bool
    vad: bool
    loopback: bool
    capture: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return self.capture and self.loopback and self.aec and self.ns and self.vad

    def to_dict(self) -> dict[str, Any]:
        return {
            "aec": self.aec,
            "ns": self.ns,
            "vad": self.vad,
            "loopback": self.loopback,
            "capture": self.capture,
            "healthy": self.healthy,
            "warnings": list(self.warnings),
        }


def evaluate_audio_frontend_readiness(
    capture_device: str | None,
    loopback_device: str | None = None,
    supports_aec: bool = False,
    supports_ns: bool = False,
    supports_vad: bool = True,
) -> AcousticFrontendReadiness:
    capture = bool(capture_device)
    loopback = bool(loopback_device)
    warnings: list[str] = []

    if not capture:
        warnings.append("capture unavailable; microphone input is blocked")
    if not loopback:
        warnings.append("loopback unavailable; speaker echo reference is degraded")
    if not supports_aec:
        warnings.append("AEC unavailable; echo cancellation is degraded")
    if not supports_ns:
        warnings.append("NS unavailable; noise suppression is degraded")
    if not supports_vad:
        warnings.append("VAD unavailable; endpointing is degraded")

    return AcousticFrontendReadiness(
        aec=supports_aec,
        ns=supports_ns,
        vad=supports_vad,
        loopback=loopback,
        capture=capture,
        warnings=warnings,
    )


@dataclass(frozen=True)
class PlaybackInterruptionPlan:
    stop_command: list[str]
    reason: str = "barge-in requires playback stop before capture continues"
    expected_max_ms: int = 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_command": list(self.stop_command),
            "reason": self.reason,
            "expected_max_ms": self.expected_max_ms,
        }


__all__ = [
    "AcousticFrontendReadiness",
    "AudioDeviceCandidate",
    "PlaybackInterruptionPlan",
    "build_aplay_command",
    "build_arecord_command",
    "evaluate_audio_frontend_readiness",
    "select_preferred_input",
]

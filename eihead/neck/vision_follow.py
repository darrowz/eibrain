"""Pan-only neck action planning for realtime visual target following."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eihead.eye.tracking import TrackingTarget


@dataclass(frozen=True)
class VisionFollowConfig:
    deadband: float = 0.15
    max_step_deg: float = 8.0
    smoothing: float = 0.5
    min_angle_delta_deg: float = 0.5
    pan_gain_deg: float = 20.0
    pan_min_deg: float = 0.0
    pan_max_deg: float = 180.0
    home_pan_deg: float = 90.0
    lost_hold_frames: int = 2
    lost_decay_step_deg: float = 2.0


@dataclass
class VisionFollowState:
    current_pan_deg: float
    last_commanded_pan_deg: float | None = None
    smoothed_error: float = 0.0
    lost_frames: int = 0


@dataclass(frozen=True)
class PanFollowAction:
    mode: str
    pan_deg: float
    pan_delta_deg: float
    tilt_deg: None = None
    reason: str = ""
    target_label: str | None = None
    target_track_id: Any | None = None
    frame_id: Any | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pan_deg": self.pan_deg,
            "pan_delta_deg": self.pan_delta_deg,
            "tilt_deg": self.tilt_deg,
            "reason": self.reason,
            "target_label": self.target_label,
            "target_track_id": self.target_track_id,
            "frame_id": self.frame_id,
        }


def plan_pan_follow_action(
    target: TrackingTarget | None,
    *,
    state: VisionFollowState,
    config: VisionFollowConfig | None = None,
) -> PanFollowAction:
    """Plan one safe horizontal-only follow step and update planner state."""

    config = config or VisionFollowConfig()
    if target is None:
        return _plan_lost_target(state=state, config=config)

    state.lost_frames = 0
    state.smoothed_error = _smooth_error(
        previous=state.smoothed_error,
        current=target.horizontal_error,
        smoothing=config.smoothing,
    )
    if abs(state.smoothed_error) < config.deadband:
        return PanFollowAction(
            mode="hold",
            pan_deg=state.current_pan_deg,
            pan_delta_deg=0.0,
            reason="deadband",
            target_label=target.label,
            target_track_id=target.track_id,
            frame_id=target.frame_id,
        )

    raw_delta = state.smoothed_error * config.pan_gain_deg
    pan_delta = _clamp(raw_delta, -config.max_step_deg, config.max_step_deg)
    next_pan = _clamp(state.current_pan_deg + pan_delta, config.pan_min_deg, config.pan_max_deg)
    pan_delta = next_pan - state.current_pan_deg
    last_commanded = state.last_commanded_pan_deg
    if last_commanded is not None and abs(next_pan - last_commanded) < config.min_angle_delta_deg:
        return PanFollowAction(
            mode="hold",
            pan_deg=state.current_pan_deg,
            pan_delta_deg=0.0,
            reason="min_angle_delta",
            target_label=target.label,
            target_track_id=target.track_id,
            frame_id=target.frame_id,
        )

    state.current_pan_deg = next_pan
    state.last_commanded_pan_deg = next_pan
    return PanFollowAction(
        mode="track",
        pan_deg=next_pan,
        pan_delta_deg=pan_delta,
        reason="tracking",
        target_label=target.label,
        target_track_id=target.track_id,
        frame_id=target.frame_id,
    )


def _plan_lost_target(*, state: VisionFollowState, config: VisionFollowConfig) -> PanFollowAction:
    state.lost_frames += 1
    state.smoothed_error = 0.0
    if state.lost_frames <= config.lost_hold_frames:
        return PanFollowAction(
            mode="hold",
            pan_deg=state.current_pan_deg,
            pan_delta_deg=0.0,
            reason="target_lost_hold",
        )

    if abs(state.current_pan_deg - config.home_pan_deg) < config.min_angle_delta_deg:
        state.current_pan_deg = config.home_pan_deg
        state.last_commanded_pan_deg = config.home_pan_deg
        return PanFollowAction(
            mode="hold",
            pan_deg=config.home_pan_deg,
            pan_delta_deg=0.0,
            reason="target_lost_home",
        )

    direction = 1.0 if config.home_pan_deg > state.current_pan_deg else -1.0
    pan_delta = direction * min(config.lost_decay_step_deg, abs(config.home_pan_deg - state.current_pan_deg))
    next_pan = _clamp(state.current_pan_deg + pan_delta, config.pan_min_deg, config.pan_max_deg)
    state.current_pan_deg = next_pan
    state.last_commanded_pan_deg = next_pan
    return PanFollowAction(
        mode="decay",
        pan_deg=next_pan,
        pan_delta_deg=pan_delta,
        reason="target_lost_decay",
    )


def _smooth_error(*, previous: float, current: float, smoothing: float) -> float:
    smoothing = _clamp(smoothing, 0.0, 1.0)
    return previous * (1.0 - smoothing) + current * smoothing


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

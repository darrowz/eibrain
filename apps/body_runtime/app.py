"""Body runtime assembly for deployable configurations."""

from __future__ import annotations

from collections import deque
import time

from eibrain.protocol.actions import PlaySpeechAction
from eibrain.body.health import DegradationManager
from eibrain.body.ear_stream import EarStreamProcessor
from eibrain.body.ear_stream import ArecordStreamCapture
from eibrain.body.organs.ear.organ import EarOrgan
from eibrain.body.organs.eye.organ import EyeOrgan
from eibrain.body.organs.mouth.organ import MouthOrgan
from eibrain.body.organs.neck.organ import NeckOrgan
from eibrain.body.sherpa_streaming import SherpaOnnxStreamingRecognizer
from eibrain.infra.config import EIBrainConfig, load_config
from eibrain.protocol.actions import Action, MoveHeadAction
from eibrain.protocol.observations import AudioTranscriptFinal


class BodyRuntimeApp:
    def __init__(self, *, config: EIBrainConfig | None = None) -> None:
        self.config = config or EIBrainConfig()
        self.organs = self._build_organs()
        self.degradation_manager = DegradationManager()
        self._recent_events: deque[dict[str, object]] = deque(maxlen=50)
        self.ear_processor: EarStreamProcessor | None = None
        self._visual_tracking_misses = 0
        self._speech_busy_until = 0.0
        self.voice_dialogue_state: dict[str, object] = {
            "enabled": False,
            "running": False,
            "phase": "idle",
            "phase_started_at_ts": time.time(),
            "turn_count": 0,
            "last_transcript": "",
            "last_reply": "",
            "last_status": "idle",
            "last_error": "",
            "last_latency_s": {},
            "last_completed_turn": {},
            "current_phase_elapsed_s": 0.0,
            "updated_at_ts": None,
        }
        self.visual_tracking_state: dict[str, object] = {
            "running": False,
            "status": "idle",
            "updated_at_ts": None,
            "frame_captured_at_ts": None,
            "detection_count": 0,
            "top_detection": None,
            "target": None,
            "last_outcome_status": None,
            "last_error": "",
            "miss_count": 0,
        }
        now_ts = time.time()
        self.interaction_state: dict[str, object] = {
            "current_mode": "sleeping",
            "reason": "idle",
            "tracking_locked": False,
            "tracking_target_label": "",
            "tracking_target_score": 0.0,
            "tracking_target_x": None,
            "tracking_raw_target_x": None,
            "tracking_stable_count": 0,
            "tracking_miss_count": 0,
            "last_attention_at_ts": None,
            "last_voice_activity_at_ts": now_ts,
            "last_neck_action_at_ts": None,
            "updated_at_ts": now_ts,
        }

    @classmethod
    def from_config_path(cls, path) -> "BodyRuntimeApp":
        return cls(config=load_config(path))

    def _build_organs(self):
        organ_configs = self.config.body.organs
        organ_types = (("ear", EarOrgan), ("eye", EyeOrgan), ("mouth", MouthOrgan), ("neck", NeckOrgan))
        if not organ_configs:
            return [organ_cls() for _, organ_cls in organ_types]
        organs = []
        for organ_name, organ_cls in organ_types:
            organ_config = organ_configs.get(organ_name)
            if organ_config is None or not organ_config.enabled:
                continue
            organs.append(organ_cls(config=organ_config))
        return organs

    def simulate_transcript(self, *, text: str, session_id: str, actor_id: str) -> AudioTranscriptFinal:
        return AudioTranscriptFinal(
            ts=1.0,
            source="ear.asr",
            text=text,
            session_id=session_id,
            actor_id=actor_id,
        )

    def _build_ear_processor(self, *, capture, recognizer) -> EarStreamProcessor:
        return EarStreamProcessor(capture=capture, recognizer=recognizer)

    def _make_capture(self, capture_cfg):
        return ArecordStreamCapture(
            device=str(capture_cfg.driver.extra.get("device", "default")),
            sample_rate=int(capture_cfg.driver.extra.get("sample_rate", 16000)),
            channels=int(capture_cfg.driver.extra.get("channels", 1)),
            streaming_vad=bool(capture_cfg.driver.extra.get("streaming_vad", False)),
            vad_frame_ms=int(capture_cfg.driver.extra.get("vad_frame_ms", 80)),
            vad_rms_threshold=float(capture_cfg.driver.extra.get("vad_rms_threshold", 0.028)),
            vad_min_voice_ms=int(capture_cfg.driver.extra.get("vad_min_voice_ms", 160)),
            vad_end_silence_ms=int(capture_cfg.driver.extra.get("vad_end_silence_ms", 360)),
            vad_pre_roll_ms=int(capture_cfg.driver.extra.get("vad_pre_roll_ms", 240)),
        )

    def _make_recognizer(self, asr_cfg):
        return SherpaOnnxStreamingRecognizer(
            model_dir=str(asr_cfg.driver.extra.get("model_dir", "")),
            model_type=str(asr_cfg.driver.extra.get("model_type", "") or "") or None,
        )

    def build_default_ear_processor(self) -> EarStreamProcessor:
        ear_cfg = self.config.body.organs.get("ear")
        if ear_cfg is None:
            raise RuntimeError("ear organ not configured")
        capture_cfg = ear_cfg.subfunctions.get("capture")
        asr_cfg = ear_cfg.subfunctions.get("asr")
        if capture_cfg is None or asr_cfg is None:
            raise RuntimeError("ear capture/asr configuration is incomplete")
        return self._build_ear_processor(
            capture=self._make_capture(capture_cfg),
            recognizer=self._make_recognizer(asr_cfg),
        )

    def transcribe_audio_window(
        self,
        *,
        chunk_count: int,
        session_id: str,
        actor_id: str,
    ) -> AudioTranscriptFinal:
        if self.is_speaking():
            return self._empty_transcript(
                session_id=session_id,
                actor_id=actor_id,
                status="speech_playback_active",
            )
        if self.ear_processor is not None:
            return self.ear_processor.transcribe_window(
                chunk_count=chunk_count,
                session_id=session_id,
                actor_id=actor_id,
            )
        ear = next((organ for organ in self.organs if organ.name == "ear"), None)
        if ear is not None and hasattr(ear, "heartbeat"):
            original_chunk_count = getattr(ear, "_chunk_count", None)
            if hasattr(ear, "_chunk_count"):
                ear._chunk_count = chunk_count
            if hasattr(ear, "_cached_heartbeat"):
                ear._cached_heartbeat = None
            try:
                heartbeat = ear.heartbeat()
            finally:
                if hasattr(ear, "_chunk_count") and original_chunk_count is not None:
                    ear._chunk_count = original_chunk_count
            asr_state = heartbeat.subfunctions.get("asr")
            capture_state = heartbeat.subfunctions.get("capture")
            details = asr_state.details if asr_state is not None else {}
            capture_details = capture_state.details if capture_state is not None else {}
            transcript = str(details.get("transcript", "") or "")
            observation = AudioTranscriptFinal(
                ts=time.time(),
                source="ear.asr",
                text=transcript,
                session_id=session_id,
                actor_id=actor_id,
            )
            self._recent_events.append(
                {
                    "kind": observation.kind,
                    "source": observation.source,
                    "status": "ok" if transcript else "degraded",
                    "session_id": session_id,
                    "recorded_at_ts": time.time(),
                    "details": {
                        "text": transcript,
                        "speech_window_summary": details.get("speech_window_summary", ""),
                        "asr_status": details.get("status"),
                        "asr_voice_activity": details.get("asr_voice_activity"),
                        "min_asr_dbfs": details.get("min_asr_dbfs"),
                        "recognizer_prewarmed": details.get("recognizer_prewarmed"),
                        "recognizer_prewarm_error": details.get("recognizer_prewarm_error"),
                        "dbfs": capture_details.get("dbfs"),
                        "rms_level": capture_details.get("rms_level"),
                        "peak_level": capture_details.get("peak_level"),
                        "payload_bytes": capture_details.get("payload_bytes"),
                        "capture_device": capture_details.get("capture_device"),
                        "sample_rate": capture_details.get("sample_rate"),
                        "channels": capture_details.get("channels"),
                        "chunk_count": capture_details.get("chunk_count"),
                        "voice_activity": capture_details.get("voice_activity"),
                        "streaming_vad": capture_details.get("streaming_vad"),
                        "vad_triggered": capture_details.get("vad_triggered"),
                        "vad_elapsed_ms": capture_details.get("vad_elapsed_ms"),
                        "captured_at_ts": capture_details.get("captured_at_ts"),
                        "asr_elapsed_ms": details.get("elapsed_ms"),
                        "capture_elapsed_ms": capture_details.get("elapsed_ms"),
                    },
                }
            )
            return observation
        if self.ear_processor is None:
            self.ear_processor = self.build_default_ear_processor()
        return self.ear_processor.transcribe_window(
            chunk_count=chunk_count,
            session_id=session_id,
            actor_id=actor_id,
        )

    def is_speaking(self) -> bool:
        return time.time() < self._speech_busy_until

    def _empty_transcript(self, *, session_id: str, actor_id: str, status: str) -> AudioTranscriptFinal:
        observation = AudioTranscriptFinal(
            ts=time.time(),
            source="ear.asr",
            text="",
            session_id=session_id,
            actor_id=actor_id,
        )
        self._recent_events.append(
            {
                "kind": observation.kind,
                "source": observation.source,
                "status": status,
                "session_id": session_id,
                "recorded_at_ts": time.time(),
                "details": {"text": "", "speech_window_summary": status},
            }
        )
        return observation

    def update_voice_dialogue_state(self, **updates: object) -> None:
        phase = updates.get("phase")
        if phase is not None and phase != self.voice_dialogue_state.get("phase"):
            updates.setdefault("phase_started_at_ts", time.time())
        self.voice_dialogue_state.update(updates)
        phase_started_at_ts = self.voice_dialogue_state.get("phase_started_at_ts")
        if isinstance(phase_started_at_ts, (int, float)):
            self.voice_dialogue_state["current_phase_elapsed_s"] = round(time.time() - float(phase_started_at_ts), 2)
        self.voice_dialogue_state["updated_at_ts"] = time.time()
        if updates.get("last_transcript") or updates.get("last_reply") or updates.get("running"):
            self._note_voice_activity()
        self._refresh_interaction_mode()

    def plan_visual_tracking_action(
        self,
        *,
        target_name: str,
        target_x: float,
        session_id: str,
        actor_id: str,
    ) -> MoveHeadAction:
        return MoveHeadAction(
            ts=1.0,
            source="eye.tracking",
            session_id=session_id,
            actor_id=actor_id,
            target_name=target_name,
            target_x=target_x,
        )

    def track_visual_target_once(
        self,
        *,
        preferred_labels: tuple[str, ...] = ("face", "person"),
        recenter_after_misses: int = 3,
        session_id: str = "tracking-session",
        actor_id: str = "vision-runtime",
    ):
        target, eye_details = self._select_visual_tracking_target(preferred_labels=preferred_labels)
        self._update_visual_tracking_state(
            running=True,
            frame_captured_at_ts=eye_details.get("frame_captured_at_ts"),
            detection_count=eye_details.get("detection_count", 0),
            top_detection=eye_details.get("top_detection"),
        )
        if target is None:
            self._visual_tracking_misses += 1
            self._note_visual_miss()
            self._update_visual_tracking_state(
                status="waiting_for_target",
                target=None,
                miss_count=self._visual_tracking_misses,
                last_outcome_status=None,
            )
            if self._visual_tracking_misses < recenter_after_misses:
                return None
            action = MoveHeadAction(
                ts=1.0,
                source="eye.tracking",
                session_id=session_id,
                actor_id=actor_id,
                target_name="recenter",
                target_angle=self._neck_home_angle(),
            )
            outcomes = self.dispatch_actions([action])
            outcome = outcomes[0] if outcomes else None
            self._update_visual_tracking_state(
                status="recentering",
                target={"label": "recenter", "target_angle": self._neck_home_angle()},
                miss_count=self._visual_tracking_misses,
                last_outcome_status=getattr(outcome, "status", None),
            )
            self._refresh_interaction_mode(force_reason="recenter_after_miss")
            return outcome
        self._visual_tracking_misses = 0
        tracking_target = self._prepare_tracking_target(target)
        if tracking_target is None:
            self._update_visual_tracking_state(
                status="holding_target",
                target=target,
                miss_count=0,
                last_outcome_status=None,
            )
            return None
        action = self.plan_visual_tracking_action(
            target_name=str(tracking_target.get("label", "target")),
            target_x=float(tracking_target["target_x"]),
            session_id=session_id,
            actor_id=actor_id,
        )
        outcomes = self.dispatch_actions([action])
        outcome = outcomes[0] if outcomes else None
        self._note_visual_target_locked(tracking_target)
        self._update_visual_tracking_state(
            status="tracking",
            target=tracking_target,
            miss_count=0,
            last_outcome_status=getattr(outcome, "status", None),
        )
        return outcome

    def snapshot(self) -> dict[str, object]:
        organ_states = [
            self._snapshot_organ(organ)
            for organ in self.organs
        ]
        degradation = self.degradation_manager.evaluate(organ_states)
        return {
            "node_id": self.config.body.node_id,
            "organ_count": len(organ_states),
            "degradation_mode": degradation.degradation_mode,
            "capabilities": degradation.capabilities.to_dict(),
            "organs": {state.organ: state.to_dict() for state in organ_states},
            "recent_event_count": len(self._recent_events),
            "voice_dialogue": dict(self.voice_dialogue_state),
            "visual_tracking": dict(self.visual_tracking_state),
            "interaction_state": dict(self.interaction_state),
        }

    def dispatch_actions(self, actions: list[Action]) -> list:
        outcomes = []
        for action in actions:
            for organ in self.organs:
                if organ.supports_action(action):
                    if isinstance(action, PlaySpeechAction):
                        self._speech_busy_until = time.time() + 120.0
                        self._note_voice_activity()
                        self._refresh_interaction_mode(force_mode="responding", force_reason="speaking")
                    try:
                        outcome = organ.handle_action(action)
                    finally:
                        if isinstance(action, PlaySpeechAction):
                            self._speech_busy_until = time.time() + 0.75
                    if outcome is not None:
                        outcomes.append(outcome)
                        self._recent_events.append(
                            {
                                "kind": outcome.kind,
                                "source": outcome.source,
                                "status": outcome.status,
                                "session_id": outcome.session_id,
                                "recorded_at_ts": time.time(),
                                "details": dict(getattr(outcome, "details", {}) or {}),
                            }
                        )
                    break
        self._refresh_interaction_mode()
        return outcomes

    def record_runtime_event(
        self,
        *,
        kind: str,
        source: str,
        status: str,
        session_id: str = "runtime",
        details: dict[str, object] | None = None,
    ) -> None:
        self._recent_events.append(
            {
                "kind": kind,
                "source": source,
                "status": status,
                "session_id": session_id,
                "recorded_at_ts": time.time(),
                "details": dict(details or {}),
            }
        )
        if source == "eye.tracking" and status == "error":
            self._update_visual_tracking_state(
                running=True,
                status="error",
                last_error=str((details or {}).get("error", "") or ""),
            )

    def _snapshot_organ(self, organ):
        if organ.name == "ear" and self.voice_dialogue_state.get("running"):
            if hasattr(organ, "passive_heartbeat"):
                return organ.passive_heartbeat()
        if organ.name == "eye" and self.voice_dialogue_state.get("running"):
            if hasattr(organ, "passive_heartbeat"):
                return organ.passive_heartbeat()
        return organ.heartbeat()

    def recent_events(self) -> list[dict[str, object]]:
        return list(self._recent_events)

    def latest_visual_frame_path(self) -> str | None:
        for organ in self.organs:
            frame_path = getattr(organ, "latest_frame_path", None)
            if isinstance(frame_path, str) and frame_path:
                return frame_path
        return None

    def _select_visual_tracking_target(
        self,
        *,
        preferred_labels: tuple[str, ...],
    ) -> tuple[dict[str, object] | None, dict[str, object]]:
        eye = next((organ for organ in self.organs if organ.name == "eye"), None)
        if eye is None:
            return None, {}
        heartbeat = eye.heartbeat()
        detection_state = heartbeat.subfunctions.get("detection")
        if detection_state is None:
            return None, {}
        detections = detection_state.details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        eye_details = {
            "frame_captured_at_ts": detection_state.details.get("frame_captured_at_ts"),
            "detection_count": len(detections),
            "top_detection": detection_state.details.get("top_detection"),
        }
        ranked: list[tuple[int, float, dict[str, object]]] = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            bbox = detection.get("bbox", {})
            if not isinstance(bbox, dict):
                continue
            try:
                x_min = float(bbox.get("x_min", 0.0))
                x_max = float(bbox.get("x_max", 0.0))
                score = float(detection.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            label = str(detection.get("label", "target"))
            priority = preferred_labels.index(label) if label in preferred_labels else len(preferred_labels)
            ranked.append(
                (
                    priority,
                    -score,
                    {
                        "label": label,
                        "score": score,
                        "target_x": max(0.0, min(1.0, (x_min + x_max) / 2.0)),
                        "bbox": bbox,
                    },
                )
            )
        if not ranked:
            return None, eye_details
        ranked.sort(key=lambda item: (item[0], item[1]))
        return ranked[0][2], eye_details

    def _neck_home_angle(self) -> int:
        neck = next((organ for organ in self.organs if organ.name == "neck"), None)
        if neck is None:
            return 90
        motor = getattr(neck.config, "subfunctions", {}).get("motor") if getattr(neck, "config", None) else None
        extra = motor.driver.extra if motor is not None else {}
        try:
            return int(extra.get("home_angle", 90))
        except (TypeError, ValueError):
            return 90

    def _update_visual_tracking_state(self, **updates: object) -> None:
        self.visual_tracking_state.update(updates)
        self.visual_tracking_state["updated_at_ts"] = time.time()
        if "last_error" not in updates and self.visual_tracking_state.get("status") != "error":
            self.visual_tracking_state["last_error"] = ""

    def _prepare_tracking_target(self, target: dict[str, object]) -> dict[str, object] | None:
        score = self._coerce_float(target.get("score"), default=0.0)
        if score < 0.3:
            self._note_visual_miss(reason="low_confidence_target")
            return None
        raw_target_x = self._coerce_float(target.get("target_x"), default=0.5)
        previous_x = self._coerce_float(self.interaction_state.get("tracking_target_x"), default=raw_target_x)
        previous_raw_x = self._coerce_float(self.interaction_state.get("tracking_raw_target_x"), default=raw_target_x)
        stable_count = int(self.interaction_state.get("tracking_stable_count", 0))
        stable_count = stable_count + 1 if abs(raw_target_x - previous_raw_x) <= 0.08 else 1
        alpha = 0.35 if stable_count > 1 else 0.65
        smoothed_target_x = raw_target_x
        if self.interaction_state.get("tracking_locked"):
            smoothed_target_x = previous_x + ((raw_target_x - previous_x) * alpha)
        current_ts = time.time()
        last_neck_action_at_ts = self.interaction_state.get("last_neck_action_at_ts")
        since_last_action_s = (
            current_ts - float(last_neck_action_at_ts)
            if isinstance(last_neck_action_at_ts, (int, float))
            else None
        )
        command_delta = abs(smoothed_target_x - previous_x)
        if since_last_action_s is not None and since_last_action_s < 0.35 and command_delta < 0.06:
            self.interaction_state.update(
                {
                    "tracking_locked": True,
                    "tracking_target_label": str(target.get("label", "target")),
                    "tracking_target_score": score,
                    "tracking_target_x": round(smoothed_target_x, 4),
                    "tracking_raw_target_x": round(raw_target_x, 4),
                    "tracking_stable_count": stable_count,
                    "tracking_miss_count": 0,
                    "updated_at_ts": current_ts,
                }
            )
            self._refresh_interaction_mode(force_mode="attention", force_reason="tracking_hold")
            return None
        return {
            **target,
            "label": str(target.get("label", "target")),
            "score": score,
            "target_x": max(0.0, min(1.0, round(smoothed_target_x, 4))),
            "raw_target_x": round(raw_target_x, 4),
            "tracking_stable_count": stable_count,
        }

    def _note_voice_activity(self) -> None:
        now_ts = time.time()
        self.interaction_state["last_voice_activity_at_ts"] = now_ts
        self.interaction_state["updated_at_ts"] = now_ts

    def _note_visual_target_locked(self, target: dict[str, object]) -> None:
        now_ts = time.time()
        self.interaction_state.update(
            {
                "tracking_locked": True,
                "tracking_target_label": str(target.get("label", "target")),
                "tracking_target_score": self._coerce_float(target.get("score"), default=0.0),
                "tracking_target_x": self._coerce_float(target.get("target_x"), default=0.5),
                "tracking_raw_target_x": self._coerce_float(target.get("raw_target_x"), default=0.5),
                "tracking_stable_count": int(target.get("tracking_stable_count", 1)),
                "tracking_miss_count": 0,
                "last_attention_at_ts": now_ts,
                "last_neck_action_at_ts": now_ts,
                "updated_at_ts": now_ts,
            }
        )
        self._refresh_interaction_mode(force_mode="attention", force_reason="visual_target_locked")

    def _note_visual_miss(self, *, reason: str = "visual_target_missing") -> None:
        miss_count = int(self.interaction_state.get("tracking_miss_count", 0)) + 1
        stable_count = int(self.interaction_state.get("tracking_stable_count", 0))
        self.interaction_state.update(
            {
                "tracking_miss_count": miss_count,
                "tracking_stable_count": max(0, stable_count - 1),
                "updated_at_ts": time.time(),
            }
        )
        if miss_count >= 3:
            self.interaction_state["tracking_locked"] = False
        self._refresh_interaction_mode(force_reason=reason)

    def _refresh_interaction_mode(
        self,
        *,
        force_mode: str | None = None,
        force_reason: str | None = None,
    ) -> None:
        now_ts = time.time()
        if force_mode is not None:
            mode = force_mode
        elif self.is_speaking():
            mode = "responding"
        else:
            last_attention_at_ts = self.interaction_state.get("last_attention_at_ts")
            attention_recent = isinstance(last_attention_at_ts, (int, float)) and now_ts - float(last_attention_at_ts) < 5.0
            if self.interaction_state.get("tracking_locked") and attention_recent:
                mode = "attention"
            elif self.voice_dialogue_state.get("running"):
                mode = "listening"
            else:
                last_voice_at_ts = self.interaction_state.get("last_voice_activity_at_ts")
                heard_recently = isinstance(last_voice_at_ts, (int, float)) and now_ts - float(last_voice_at_ts) < 8.0
                mode = "listening" if heard_recently else "sleeping"
        self.interaction_state["current_mode"] = mode
        self.interaction_state["reason"] = force_reason or self._interaction_reason(mode)
        self.interaction_state["updated_at_ts"] = now_ts

    def _interaction_reason(self, mode: str) -> str:
        if mode == "responding":
            return "speaking"
        if mode == "attention":
            return "visual_target_locked"
        if mode == "listening":
            return "voice_runtime_active"
        return "idle"

    @staticmethod
    def _coerce_float(value: object, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

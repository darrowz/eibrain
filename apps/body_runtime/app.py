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

    @classmethod
    def from_config_path(cls, path) -> "BodyRuntimeApp":
        return cls(config=load_config(path))

    def _build_organs(self):
        organ_configs = self.config.body.organs
        return [
            EarOrgan(config=organ_configs.get("ear")),
            EyeOrgan(config=organ_configs.get("eye")),
            MouthOrgan(config=organ_configs.get("mouth")),
            NeckOrgan(config=organ_configs.get("neck")),
        ]

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
                        "dbfs": capture_details.get("dbfs"),
                        "rms_level": capture_details.get("rms_level"),
                        "peak_level": capture_details.get("peak_level"),
                        "payload_bytes": capture_details.get("payload_bytes"),
                        "capture_device": capture_details.get("capture_device"),
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
        target = self._select_visual_tracking_target(preferred_labels=preferred_labels)
        if target is None:
            self._visual_tracking_misses += 1
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
            return outcomes[0] if outcomes else None
        self._visual_tracking_misses = 0
        action = self.plan_visual_tracking_action(
            target_name=str(target.get("label", "target")),
            target_x=float(target["target_x"]),
            session_id=session_id,
            actor_id=actor_id,
        )
        outcomes = self.dispatch_actions([action])
        return outcomes[0] if outcomes else None

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
        }

    def dispatch_actions(self, actions: list[Action]) -> list:
        outcomes = []
        for action in actions:
            for organ in self.organs:
                if organ.supports_action(action):
                    if isinstance(action, PlaySpeechAction):
                        self._speech_busy_until = time.time() + 120.0
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
        return outcomes

    def _snapshot_organ(self, organ):
        if organ.name in {"ear", "mouth", "neck"} and self.voice_dialogue_state.get("running"):
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
    ) -> dict[str, object] | None:
        eye = next((organ for organ in self.organs if organ.name == "eye"), None)
        if eye is None:
            return None
        heartbeat = eye.heartbeat()
        detection_state = heartbeat.subfunctions.get("detection")
        if detection_state is None:
            return None
        detections = detection_state.details.get("detections", [])
        if not isinstance(detections, list):
            return None
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
            return None
        ranked.sort(key=lambda item: (item[0], item[1]))
        return ranked[0][2]

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

"""Body runtime assembly for deployable configurations."""

from __future__ import annotations

from collections import deque
import time

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
        if self.ear_processor is None:
            self.ear_processor = self.build_default_ear_processor()
        return self.ear_processor.transcribe_window(
            chunk_count=chunk_count,
            session_id=session_id,
            actor_id=actor_id,
        )

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

    def snapshot(self) -> dict[str, object]:
        organ_states = [organ.heartbeat() for organ in self.organs]
        degradation = self.degradation_manager.evaluate(organ_states)
        return {
            "node_id": self.config.body.node_id,
            "organ_count": len(organ_states),
            "degradation_mode": degradation.degradation_mode,
            "capabilities": degradation.capabilities.to_dict(),
            "organs": {state.organ: state.to_dict() for state in organ_states},
            "recent_event_count": len(self._recent_events),
        }

    def dispatch_actions(self, actions: list[Action]) -> list:
        outcomes = []
        for action in actions:
            for organ in self.organs:
                if organ.supports_action(action):
                    outcome = organ.handle_action(action)
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

    def recent_events(self) -> list[dict[str, object]]:
        return list(self._recent_events)

    def latest_visual_frame_path(self) -> str | None:
        for organ in self.organs:
            frame_path = getattr(organ, "latest_frame_path", None)
            if isinstance(frame_path, str) and frame_path:
                return frame_path
        return None

"""Mouth organ implementation."""

from __future__ import annotations

import time

from eibrain.body.health.organ_health import OrganHealth, SubfunctionHealth
from eibrain.body.organs.base import BaseOrgan
from eibrain.protocol.actions import PlaySpeechAction, StopSpeechAction
from eibrain.protocol.outcomes import ActionExecuted, SpeechPlaybackCompleted


class MouthOrgan(BaseOrgan):
    name = "mouth"
    subfunction_names = ("tts_plan", "tts_playback")

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)
        self._last_plan: dict[str, object] | None = None
        self._last_playback: dict[str, object] | None = None

    def passive_heartbeat(self) -> OrganHealth:
        plan_details = {"driver": self._driver_kind("tts_plan"), "status": "live_probe_skipped"}
        playback_details = {"driver": self._driver_kind("tts_playback"), "status": "live_probe_skipped"}
        plan_details.update(self._voice_config_details())
        playback_details.update(self._voice_config_details())
        if self._last_plan is not None:
            plan_details.update(self._last_plan)
        if self._last_playback is not None:
            playback_details.update(self._last_playback)
        subfunctions = {
            "tts_plan": SubfunctionHealth(name="tts_plan", health="healthy", details=plan_details),
            "tts_playback": SubfunctionHealth(name="tts_playback", health="healthy", details=playback_details),
        }
        return OrganHealth(organ=self.name, health="healthy", subfunctions=subfunctions)

    def heartbeat(self) -> OrganHealth:
        if self._driver_kind("tts_plan") == "noop" and self._driver_kind("tts_playback") == "noop":
            return super().heartbeat()

        plan_state = self._tts_plan_health()
        playback_state = self._tts_playback_health()
        subfunctions = {
            "tts_plan": plan_state,
            "tts_playback": playback_state,
        }
        statuses = [state.health for state in subfunctions.values()]
        if statuses and all(status == "healthy" for status in statuses):
            health = "healthy"
        elif any(status == "healthy" for status in statuses) or any(status == "degraded" for status in statuses):
            health = "degraded"
        else:
            health = "unavailable"
        return OrganHealth(organ=self.name, health=health, subfunctions=subfunctions)

    def supports_action(self, action) -> bool:
        return isinstance(action, (PlaySpeechAction, StopSpeechAction))

    def handle_action(self, action):
        if isinstance(action, PlaySpeechAction):
            self._last_plan = {
                "status": "planned",
                "text_preview": self._preview_text(action.text),
                "text_char_count": len(action.text),
                "planned_at_ts": action.ts or time.time(),
            }
            result = self.drivers["tts_playback"].invoke("play_speech", {"text": action.text})
            self._last_playback = {
                "status": result.status,
                "text_preview": self._preview_text(action.text),
                "text_char_count": len(action.text),
                "played_at_ts": action.ts or time.time(),
                "details": dict(result.details),
            }
            return SpeechPlaybackCompleted(
                ts=action.ts,
                source="mouth.tts_playback",
                status=result.status,
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
            )
        if isinstance(action, StopSpeechAction):
            result = self.drivers["tts_playback"].invoke("stop_speech", {})
            return ActionExecuted(
                ts=action.ts,
                source="mouth.tts_playback",
                session_id=action.session_id,
                actor_id=action.actor_id,
                target_id=action.target_id,
                action_kind=action.kind,
                details=result.details,
            )
        return None

    def _tts_plan_health(self) -> SubfunctionHealth:
        if self._driver_kind("tts_plan") == "noop":
            return self._subfunction_health("tts_plan")
        probe = self.drivers["tts_plan"].heartbeat()
        details = self._merge_probe_details(dict(probe.details))
        details.update(self._voice_config_details())
        if self._last_plan is not None:
            details.update(self._last_plan)
            details["status"] = self._last_plan.get("status", "planned")
        else:
            details["status"] = "ready"
        return SubfunctionHealth(
            name="tts_plan",
            health=self._normalize_status(probe.status),
            details=details,
        )

    def _tts_playback_health(self) -> SubfunctionHealth:
        if self._driver_kind("tts_playback") == "noop":
            return self._subfunction_health("tts_playback")
        probe = self.drivers["tts_playback"].heartbeat()
        details = self._merge_probe_details(dict(probe.details))
        details.update(self._voice_config_details())
        health = self._normalize_status(probe.status)
        if self._last_playback is not None:
            playback_details = dict(self._last_playback.get("details", {}))
            if isinstance(playback_details, dict):
                details.update(playback_details)
            details["text_preview"] = self._last_playback.get("text_preview", "")
            details["text_char_count"] = self._last_playback.get("text_char_count", 0)
            details["played_at_ts"] = self._last_playback.get("played_at_ts")
            details["status"] = self._last_playback.get("status", probe.status)
            health = self._normalize_status(str(self._last_playback.get("status", probe.status)))
        else:
            details["status"] = "ready"
        return SubfunctionHealth(
            name="tts_playback",
            health=health,
            details=details,
        )

    def _driver_kind(self, name: str) -> str:
        config = self.config.subfunctions.get(name)
        if config is None:
            return "noop"
        return str(config.driver.kind)

    def _voice_config_details(self) -> dict[str, object]:
        playback_cfg = self.config.subfunctions.get("tts_playback")
        if playback_cfg is None:
            return {}
        extra = playback_cfg.driver.extra
        return {
            "backend": extra.get("backend", "espeak"),
            "voice_id": extra.get("voice_id"),
            "model": extra.get("model"),
            "output_device": extra.get("output_device"),
        }

    @staticmethod
    def _merge_probe_details(probe: dict[str, object]) -> dict[str, object]:
        merged = dict(probe)
        merged["driver"] = merged.get("driver", "command")
        nested = merged.get("details", {})
        if not isinstance(nested, dict):
            nested = {}
        merged["details"] = nested
        return merged

    @staticmethod
    def _preview_text(text: str) -> str:
        collapsed = " ".join(text.split())
        return collapsed[:80]

"""Body degradation manager."""

from __future__ import annotations

from dataclasses import dataclass

from .capability_matrix import CapabilityMatrix
from .organ_health import OrganHealth


@dataclass(slots=True)
class DegradationResult:
    capabilities: CapabilityMatrix
    degradation_mode: str


class DegradationManager:
    def evaluate(self, organ_states: list[OrganHealth]) -> DegradationResult:
        by_name = {state.organ: state for state in organ_states}

        ear = by_name.get("ear")
        mouth = by_name.get("mouth")
        eye = by_name.get("eye")
        neck = by_name.get("neck")

        capabilities = CapabilityMatrix(
            can_hear_voice=self._is_real_capability(ear, "capture", allow_degraded=True),
            can_transcribe_speech=self._is_real_capability(ear, "asr"),
            can_see_people=self._is_real_capability(eye, "detection", allow_degraded=True),
            can_identify_person=self._is_real_capability(eye, "identity"),
            can_speak=self._is_real_capability(mouth, "tts_playback", allow_degraded=True),
            can_orient_head=self._is_real_capability(neck, "motor", allow_degraded=True),
        )

        degradation_mode = "normal"
        if capabilities.can_hear_voice and not capabilities.can_transcribe_speech:
            degradation_mode = "low_confidence_body"
        elif not capabilities.can_speak:
            degradation_mode = "mute_companion"
        elif not capabilities.can_orient_head:
            degradation_mode = "fixed_gaze"

        return DegradationResult(capabilities=capabilities, degradation_mode=degradation_mode)

    @staticmethod
    def _is_real_capability(
        organ: OrganHealth | None,
        subfunction_name: str,
        *,
        allow_degraded: bool = False,
    ) -> bool:
        if organ is None:
            return False
        subfunction = organ.subfunctions.get(subfunction_name)
        if subfunction is None:
            return False
        if subfunction.details.get("driver") == "noop":
            return False
        if subfunction.health == "healthy":
            return True
        if allow_degraded and subfunction.health == "degraded":
            return True
        return False

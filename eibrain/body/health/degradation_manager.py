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
            can_hear_voice=bool(ear and ear.subfunctions.get("capture") and ear.subfunctions["capture"].health != "unavailable"),
            can_transcribe_speech=bool(ear and ear.subfunctions.get("asr") and ear.subfunctions["asr"].health == "healthy"),
            can_see_people=bool(eye and eye.subfunctions.get("detection") and eye.subfunctions["detection"].health != "unavailable"),
            can_identify_person=bool(eye and eye.subfunctions.get("identity") and eye.subfunctions["identity"].health == "healthy"),
            can_speak=bool(mouth and mouth.subfunctions.get("tts_playback") and mouth.subfunctions["tts_playback"].health != "unavailable"),
            can_orient_head=bool(neck and neck.subfunctions.get("motor") and neck.subfunctions["motor"].health != "unavailable"),
        )

        degradation_mode = "normal"
        if capabilities.can_hear_voice and not capabilities.can_transcribe_speech:
            degradation_mode = "low_confidence_body"
        elif not capabilities.can_speak:
            degradation_mode = "mute_companion"
        elif not capabilities.can_orient_head:
            degradation_mode = "fixed_gaze"

        return DegradationResult(capabilities=capabilities, degradation_mode=degradation_mode)

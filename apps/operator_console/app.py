"""Operator console placeholder."""

from __future__ import annotations

import time


class OperatorConsoleApp:
    """Operator console for status summaries."""

    IMPORTANT_CAPABILITIES = (
        "can_hear_voice",
        "can_transcribe_speech",
        "can_see_people",
        "can_speak",
        "can_orient_head",
    )

    def build_status_report(
        self,
        *,
        body_snapshot: dict[str, object],
        cognitive_snapshot: dict[str, object],
        traces: list[dict[str, object]],
    ) -> dict[str, object]:
        degradation = str(body_snapshot.get("degradation_mode", "unknown"))
        capabilities = dict(body_snapshot.get("capabilities", {}))
        organs = dict(body_snapshot.get("organs", {}))
        warnings = [
            f"{name}=false"
            for name in self.IMPORTANT_CAPABILITIES
            if capabilities.get(name) is False
        ]
        degraded_organs = sorted(
            organ_name
            for organ_name, snapshot in organs.items()
            if isinstance(snapshot, dict) and snapshot.get("health") != "healthy"
        )
        system_health = "healthy" if degradation == "normal" and not warnings and not degraded_organs else "degraded"
        return {
            "system_health": system_health,
            "generated_at_ts": time.time(),
            "trace_count": len(traces),
            "warnings": warnings,
            "degraded_organs": degraded_organs,
            "body": body_snapshot,
            "cognition": cognitive_snapshot,
            "recent_traces": traces[-5:],
        }

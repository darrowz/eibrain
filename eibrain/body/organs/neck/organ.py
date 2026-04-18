"""Neck organ implementation."""

from __future__ import annotations

from eibrain.body.organs.base import BaseOrgan
from eibrain.body.runtime_linux import map_target_x_to_angle
from eibrain.protocol.actions import MoveHeadAction
from eibrain.protocol.outcomes import ActionExecuted


class NeckOrgan(BaseOrgan):
    name = "neck"
    subfunction_names = ("motor", "tracking")

    def supports_action(self, action) -> bool:
        return isinstance(action, MoveHeadAction)

    def handle_action(self, action):
        if not isinstance(action, MoveHeadAction):
            return None
        config = self.config.subfunctions.get("motor")
        extra = config.driver.extra if config is not None else {}
        pan_min = int(extra.get("pan_min", 40))
        pan_max = int(extra.get("pan_max", 140))
        target_angle = action.target_angle
        if target_angle is None and action.target_x is not None:
            target_angle = map_target_x_to_angle(target_x=action.target_x, pan_min=pan_min, pan_max=pan_max)
        result = self.drivers["motor"].invoke(
            "move_head",
            {
                "target_id": action.target_id,
                "target_name": action.target_name,
                "target_x": action.target_x,
                "target_angle": target_angle,
            },
        )
        return ActionExecuted(
            ts=action.ts,
            source="neck.motor",
            status=result.status,
            session_id=action.session_id,
            actor_id=action.actor_id,
            target_id=action.target_id,
            action_kind=action.kind,
            details=result.details,
        )

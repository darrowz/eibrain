from __future__ import annotations


def test_map_target_x_to_angle_maps_frame_edges() -> None:
    from eibrain.body.runtime_linux import map_target_x_to_angle

    assert map_target_x_to_angle(target_x=0.0, pan_min=40, pan_max=140) == 40
    assert map_target_x_to_angle(target_x=0.5, pan_min=40, pan_max=140) == 90
    assert map_target_x_to_angle(target_x=1.0, pan_min=40, pan_max=140) == 140


def test_neck_uses_target_x_to_compute_angle() -> None:
    from eibrain.body.organs.neck.organ import NeckOrgan
    from eibrain.infra.config import DriverConfig, OrganConfig, SubfunctionConfig
    from eibrain.protocol.actions import MoveHeadAction

    class _Driver:
        def heartbeat(self):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(status="healthy", details={"driver": "fake"})

        def invoke(self, operation: str, payload: dict[str, object]):
            from eibrain.body.drivers.base import DriverResult

            return DriverResult(status="ok", details={"operation": operation, "payload": payload})

    organ = NeckOrgan(
        config=OrganConfig(
            enabled=True,
            subfunctions={
                "motor": SubfunctionConfig(
                    driver=DriverConfig(
                        kind="noop",
                        extra={"pan_min": 40, "pan_max": 140},
                    )
                ),
                "tracking": SubfunctionConfig(),
            },
        )
    )
    organ.drivers["motor"] = _Driver()

    outcome = organ.handle_action(
        MoveHeadAction(
            ts=1.0,
            source="test",
            target_id="user-1",
            target_name="speaker",
            target_x=0.25,
        )
    )

    assert outcome.details["payload"]["target_angle"] == 65

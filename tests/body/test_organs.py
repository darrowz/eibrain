from __future__ import annotations


def test_phase1_organs_publish_heartbeat_and_health() -> None:
    from eibrain.body.organs.ear.organ import EarOrgan
    from eibrain.body.organs.eye.organ import EyeOrgan
    from eibrain.body.organs.mouth.organ import MouthOrgan
    from eibrain.body.organs.neck.organ import NeckOrgan

    organs = [EarOrgan(), EyeOrgan(), MouthOrgan(), NeckOrgan()]

    for organ in organs:
        heartbeat = organ.heartbeat()
        assert heartbeat.organ == organ.name
        assert heartbeat.health in {"healthy", "degraded", "unavailable"}
        assert heartbeat.subfunctions



def test_base_organ_preserves_waiting_status_as_healthy_heartbeat() -> None:
    from eibrain.body.organs.base import BaseOrgan
    from eibrain.body.drivers.base import DriverResult
    from eibrain.infra.config import OrganConfig, SubfunctionConfig

    class _Driver:
        def heartbeat(self):
            return DriverResult(status="waiting_for_data", details={"status": "waiting_for_data"})

    class _Organ(BaseOrgan):
        name = "stub"
        subfunction_names = ("capture",)

        def __init__(self):
            super().__init__(config=OrganConfig(subfunctions={"capture": SubfunctionConfig()}))
            self.drivers = {"capture": _Driver()}

    heartbeat = _Organ().heartbeat()

    assert heartbeat.health == "healthy"
    assert heartbeat.subfunctions["capture"].health == "healthy"

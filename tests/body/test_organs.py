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

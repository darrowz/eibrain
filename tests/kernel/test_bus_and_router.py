from __future__ import annotations


def test_kernel_bus_publishes_to_subscribers() -> None:
    from eibrain.kernel.bus import KernelBus
    from eibrain.protocol.envelopes import Envelope

    seen: list[Envelope] = []
    bus = KernelBus()
    bus.subscribe("observations", seen.append)

    envelope = Envelope(channel="observations", payload={"kind": "x"})
    bus.publish(envelope)

    assert seen == [envelope]


def test_router_and_guard_process_envelope() -> None:
    from eibrain.kernel.guards import AllowAllGuard
    from eibrain.kernel.router import EnvelopeRouter
    from eibrain.protocol.envelopes import Envelope

    routed: list[Envelope] = []
    router = EnvelopeRouter(guards=[AllowAllGuard()])
    router.register("actions", routed.append)

    envelope = Envelope(channel="actions", payload={"kind": "play_speech_action"})
    accepted = router.route(envelope)

    assert accepted is True
    assert routed == [envelope]

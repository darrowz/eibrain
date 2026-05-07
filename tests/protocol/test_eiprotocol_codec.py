from __future__ import annotations

import json
from typing import Any

import pytest

from eiprotocol.models import EventEnvelope, SourceRef


def _event_payload(**overrides: Any) -> dict[str, Any]:
    event = EventEnvelope(
        event_id="evt_codec_001",
        event_type="capability",
        name="ei.capability.manifest.report",
        time="2026-05-04T10:31:00+08:00",
        sequence=1,
        request_id="req_codec_001",
        source=SourceRef(domain="eihead", instance_id="honjia", device_id="honjia"),
        content={
            "manifestId": "cap_honjia_001",
            "manifestVersion": "2026-05-04",
            "capabilities": [],
            "displayName": "红家",
        },
    )
    payload = event.to_dict()
    payload.update(overrides)
    return payload


class EventLike:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


def test_dumps_and_loads_event_round_trip() -> None:
    from eiprotocol.codec import dumps_event, loads_event

    payload = _event_payload()

    restored = loads_event(dumps_event(payload))

    assert isinstance(restored, EventEnvelope)
    assert restored.to_dict() == payload


def test_event_envelope_from_dict_accepts_legacy_snake_case_ttl_ms() -> None:
    payload = _event_payload(ttlMs=None, ttl_ms=2500)

    restored = EventEnvelope.from_dict(payload)

    assert restored.ttl_ms == 2500
    assert restored.to_dict()["ttlMs"] == 2500
    assert "ttl_ms" not in restored.to_dict()


def test_loads_event_accepts_utf8_bytes() -> None:
    from eiprotocol.codec import dumps_event, loads_event

    text = dumps_event(_event_payload()).encode("utf-8")

    restored = loads_event(text)

    assert restored.event_id == "evt_codec_001"
    assert restored.content["displayName"] == "红家"


def test_canonical_json_is_stable_for_equivalent_dicts() -> None:
    from eiprotocol.codec import canonical_event_json, dumps_event

    payload = _event_payload(content={"z": 1, "a": {"beta": 2, "alpha": "红家"}})
    reordered = dict(reversed(list(payload.items())))

    canonical = canonical_event_json(payload)

    assert canonical == canonical_event_json(reordered)
    assert canonical == dumps_event(reordered, canonical=True)
    assert '"alpha":"红家"' in canonical
    assert ", " not in canonical
    assert ": " not in canonical


def test_loads_event_rejects_invalid_json() -> None:
    from eiprotocol.codec import EventDecodeError, loads_event

    with pytest.raises(EventDecodeError) as exc_info:
        loads_event("{not-json")

    assert exc_info.value.kind == "invalid_json"
    assert exc_info.value.to_dict()["message"]


def test_loads_event_rejects_non_object_json() -> None:
    from eiprotocol.codec import EventDecodeError, loads_event

    with pytest.raises(EventDecodeError) as exc_info:
        loads_event("[]")

    assert exc_info.value.kind == "invalid_json_object"


def test_loads_event_rejects_invalid_utf8_bytes() -> None:
    from eiprotocol.codec import EventDecodeError, loads_event

    with pytest.raises(EventDecodeError) as exc_info:
        loads_event(b"\xff")

    assert exc_info.value.kind == "invalid_encoding"


def test_loads_event_rejects_invalid_event_when_validation_enabled() -> None:
    from eiprotocol.codec import EventDecodeError, loads_event

    payload = _event_payload()
    del payload["source"]

    with pytest.raises(EventDecodeError) as exc_info:
        loads_event(json.dumps(payload))

    error = exc_info.value
    assert error.kind == "invalid_event"
    assert "source is required" in error.details["errors"]
    assert error.to_dict() == {
        "kind": "invalid_event",
        "message": "event failed validation",
        "details": {"errors": ["source is required"]},
    }


def test_loads_event_rejects_known_event_missing_catalog_content_fields() -> None:
    from eiprotocol.codec import EventDecodeError, loads_event

    payload = _event_payload()
    del payload["content"]["manifestVersion"]

    with pytest.raises(EventDecodeError) as exc_info:
        loads_event(json.dumps(payload))

    error = exc_info.value
    assert error.kind == "invalid_event"
    assert any("missing_content_field at content.manifestVersion" in error for error in error.details["errors"])


def test_event_to_dict_accepts_event_like_to_dict_input() -> None:
    from eiprotocol.codec import dumps_event, event_to_dict, loads_event

    payload = _event_payload()
    event_like = EventLike(payload)

    assert event_to_dict(event_like) == payload
    assert loads_event(dumps_event(event_like)).to_dict() == payload

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "eiprotocol"


def _assert_strict_round_trip_and_route(event: Any) -> dict[str, Any]:
    from eiprotocol.codec import dumps_event, loads_event
    from eiprotocol.event_routing import classify_event
    from eiprotocol.validation import validate_event_strict

    payload = event.to_dict()

    assert validate_event_strict(payload, known_event_required=True) == []
    assert loads_event(dumps_event(event)).to_dict() == payload
    route = classify_event(event)
    assert route["status"] == "routed"
    assert route["route"] == "realtime_vision_frame"
    return payload


def test_realtime_vision_observation_typed_roundtrip_stabilizes_list_and_dict_bbox() -> None:
    from eiprotocol import Detection, RealtimeVisionObservation

    observation = RealtimeVisionObservation(
        frame_id="frame-typed-1",
        width=1280,
        height=720,
        frame_age_ms=24.5,
        backend="hailo",
        detections=[
            Detection(
                label="person",
                score=0.91,
                bbox={"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6},
                track_id="track-person-1",
            ),
            Detection(
                label="cat",
                score=0.73,
                bbox=[0.6, 0.3, 0.2, 0.2],
                track_id="track-cat-1",
            ),
        ],
        boxes=[
            {"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6},
            [0.6, 0.3, 0.2, 0.2],
        ],
        scores=[0.91, 0.73],
        tracked_target={"label": "person", "trackId": "track-person-1"},
        latency_ms={"capture": 8.0, "detect": 15.0},
        metadata={"camera": "front"},
    )

    content = observation.to_content()
    restored = RealtimeVisionObservation.from_content(content)

    assert content["detections"][0]["bbox"] == {"x": 0.2, "y": 0.1, "w": 0.4, "h": 0.6}
    assert content["detections"][1]["bbox"] == [0.6, 0.3, 0.2, 0.2]
    assert restored.to_content() == content


def test_realtime_vision_fixture_is_strict_valid_and_typed_roundtrips() -> None:
    from eiprotocol import RealtimeVisionObservation, require_event_definition
    from eiprotocol.codec import dumps_event, loads_event
    from eiprotocol.event_routing import classify_event
    from eiprotocol.validation import validate_event_strict

    payload = json.loads((FIXTURE_DIR / "realtime_vision_frame.json").read_text(encoding="utf-8"))

    definition = require_event_definition("ei.observation.vision.frame")
    assert definition.direction == "head_to_brain"
    assert validate_event_strict(payload, known_event_required=True) == []
    assert loads_event(dumps_event(payload)).to_dict() == payload
    assert classify_event(payload)["route"] == "realtime_vision_frame"
    assert RealtimeVisionObservation.from_content(payload["content"]).to_content() == payload["content"]


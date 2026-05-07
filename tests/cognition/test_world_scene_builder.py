from __future__ import annotations

from eibrain.cognition.world.scene_builder import (
    build_scene_graph,
    should_write_world_observation,
)


def _state(*detections: dict[str, object]) -> dict[str, object]:
    return {
        "backend": "hailo8",
        "frame_id": "frame-1",
        "detections": list(detections),
    }


def _det(label: str, bbox: dict[str, float], confidence: float = 0.9) -> dict[str, object]:
    return {
        "label": label,
        "confidence": confidence,
        "bbox": bbox,
        "model_id": "yolov8n-coco80",
    }


def test_build_scene_graph_marks_new_object_as_appeared() -> None:
    scene = build_scene_graph(
        _state(_det("cup", {"x_min": 0.10, "y_min": 0.20, "x_max": 0.20, "y_max": 0.40}))
    )

    assert scene["objects"][0]["label"] == "cup"
    assert scene["objects"][0]["stable_id"].startswith("cup:")
    assert scene["objects"][0]["region"] == "left_middle"
    assert scene["events"] == [
        {
            "type": "appeared",
            "object_id": scene["objects"][0]["stable_id"],
            "label": "cup",
        }
    ]
    assert scene["scene_id"].startswith("scene:")
    assert scene["dedupe_key"].startswith("world_scene:")
    assert scene["fingerprint"]
    assert "Observed cup" in scene["summary"]


def test_build_scene_graph_marks_missing_object_as_disappeared() -> None:
    previous = build_scene_graph(
        _state(_det("cup", {"x_min": 0.10, "y_min": 0.20, "x_max": 0.20, "y_max": 0.40}))
    )

    scene = build_scene_graph(_state(), previous_scene=previous)

    assert scene["events"] == [
        {
            "type": "disappeared",
            "object_id": previous["objects"][0]["stable_id"],
            "label": "cup",
        }
    ]


def test_build_scene_graph_marks_object_moved() -> None:
    previous = build_scene_graph(
        _state(_det("cup", {"x_min": 0.10, "y_min": 0.20, "x_max": 0.20, "y_max": 0.40}))
    )

    scene = build_scene_graph(
        _state(_det("cup", {"x_min": 0.45, "y_min": 0.20, "x_max": 0.55, "y_max": 0.40})),
        previous_scene=previous,
    )

    assert scene["events"][0]["type"] == "moved"
    assert scene["events"][0]["label"] == "cup"
    assert scene["events"][0]["from_region"] == "left_middle"
    assert scene["events"][0]["to_region"] == "center_middle"


def test_build_scene_graph_generates_spatial_relations() -> None:
    scene = build_scene_graph(
        _state(
            _det("person", {"x_min": 0.10, "y_min": 0.10, "x_max": 0.25, "y_max": 0.35}),
            _det("cup", {"x_min": 0.70, "y_min": 0.12, "x_max": 0.80, "y_max": 0.30}),
            _det("book", {"x_min": 0.12, "y_min": 0.70, "x_max": 0.28, "y_max": 0.85}),
        )
    )

    relation_types = {
        (item["subject_label"], item["relation"], item["object_label"])
        for item in scene["relations"]
    }

    assert ("person", "left_of", "cup") in relation_types
    assert ("cup", "right_of", "person") in relation_types
    assert ("person", "above", "book") in relation_types
    assert ("book", "below", "person") in relation_types
    assert ("person", "near", "book") in relation_types


def test_world_scene_graph_preserves_multimodal_features_and_extracts_attention_events() -> None:
    scene = build_scene_graph(
        _state(
            {
                "label": "person",
                "confidence": 0.94,
                "bbox": {"x_min": 0.20, "y_min": 0.18, "x_max": 0.58, "y_max": 0.92},
                "stable_id": "person-001",
                "pose": {
                    "keypoints": [
                        {"name": "right_wrist", "x": 0.61, "y": 0.58, "confidence": 0.76},
                    ]
                },
                "clip_labels": ["person", {"label": "operator", "confidence": 0.62}],
                "semantic_labels": ["human"],
                "depth_m": 0.72,
                "distance_band": "near",
                "looking_at_device": True,
                "source": "hailo",
                "model_id": "pose-clip-placeholder",
                "provenance": {"device": "hailo8l"},
            },
            {
                "label": "phone",
                "confidence": 0.84,
                "bbox": {"x_min": 0.58, "y_min": 0.52, "x_max": 0.68, "y_max": 0.66},
                "stable_id": "phone-001",
                "semantic_labels": ["device"],
                "depth_m": 0.78,
            },
        )
    )

    person = next(item for item in scene["objects"] if item["stable_id"] == "person-001")
    event_types = {event["type"] for event in scene["events"]}
    relation_types = {
        (item["subject_id"], item["relation"], item["object_id"])
        for item in scene["relations"]
    }

    assert person["pose"]["keypoints"] == [{"name": "right_wrist", "x": 0.61, "y": 0.58, "confidence": 0.76}]
    assert person["clip_labels"] == [{"label": "person"}, {"label": "operator", "confidence": 0.62}]
    assert person["semantic_labels"] == ["human"]
    assert person["depth_m"] == 0.72
    assert person["distance_band"] == "near"
    assert person["source"] == "hailo"
    assert person["model_id"] == "pose-clip-placeholder"
    assert person["provenance"]["device"] == "hailo8l"
    assert "looking_at_device" in event_types
    assert "hand_near_object" in event_types
    assert ("person-001", "hand_near_object", "phone-001") in relation_types


def test_world_scene_graph_accepts_protocol_bbox_depth_and_distance_objects() -> None:
    scene = build_scene_graph(
        _state(
            {
                "label": "person",
                "confidence": 0.94,
                "bbox": {"x": 0.20, "y": 0.18, "w": 0.38, "h": 0.74},
                "stable_id": "person-001",
                "clipLabels": [{"label": "person at desk", "score": 0.84}],
                "semanticLabels": [{"label": "workspace", "confidence": 0.79}],
                "depth": {"median": 0.72, "unit": "m"},
                "distance": {"fromCameraM": 0.72},
            },
        )
    )

    person = scene["objects"][0]

    assert person["bbox"] == {"x_min": 0.2, "y_min": 0.18, "x_max": 0.58, "y_max": 0.92}
    assert person["depth_m"] == 0.72
    assert person["distance_band"] == "near"
    assert person["clip_labels"][0]["label"] == "person at desk"
    assert person["semantic_labels"][0] == "workspace"


def test_world_scene_graph_accepts_protocol_list_bbox_and_tracking_diagnostics() -> None:
    scene = build_scene_graph(
        _state(
            {
                "label": "person",
                "score": 0.91,
                "bbox": [0.62, 0.35, 0.12, 0.18],
                "trackingDiagnostics": {
                    "trackIdSwitchCount": 0,
                    "targetStabilityRatio": 1.0,
                },
            },
        )
    )

    person = scene["objects"][0]

    assert person["bbox"] == {"x_min": 0.62, "y_min": 0.35, "x_max": 0.74, "y_max": 0.53}
    assert person["tracking_diagnostics"]["trackIdSwitchCount"] == 0


def test_should_write_world_observation_skips_unchanged_scene() -> None:
    previous = build_scene_graph(
        _state(_det("cup", {"x_min": 0.10, "y_min": 0.20, "x_max": 0.20, "y_max": 0.40}))
    )
    current = build_scene_graph(
        _state(_det("cup", {"x_min": 0.11, "y_min": 0.20, "x_max": 0.21, "y_max": 0.40})),
        previous_scene=previous,
    )

    assert should_write_world_observation(current, previous) is False
    assert should_write_world_observation(
        build_scene_graph(
            _state(_det("person", {"x_min": 0.70, "y_min": 0.20, "x_max": 0.80, "y_max": 0.50})),
            previous_scene=previous,
        ),
        previous,
    ) is True


def test_cognitive_runtime_payload_includes_world_scene_graph() -> None:
    from apps.cognitive_runtime.app import CognitiveRuntimeApp

    runtime = CognitiveRuntimeApp()
    payload = runtime.build_world_observation_payload(
        _state(
            _det("person", {"x_min": 0.10, "y_min": 0.10, "x_max": 0.25, "y_max": 0.35}),
            _det("cup", {"x_min": 0.70, "y_min": 0.12, "x_max": 0.80, "y_max": 0.30}),
        ),
        session_id="vision:desk",
    )

    scene = payload["content"]["scene"]
    assert scene["objects"][0]["stable_id"]
    assert scene["relations"]
    assert scene["events"][0]["type"] == "appeared"
    assert payload["content"]["events"] == scene["events"]
    assert payload["content"]["spatial"]["relations"] == scene["relations"]
    assert scene["dedupe_key"].startswith("world_scene:")
    assert payload["meta"]["dedupe_key"].startswith("world_observation:")
    assert payload["meta"]["scene_dedupe_key"] == scene["dedupe_key"]
    assert payload["meta"]["scene_id"] == scene["scene_id"]
    assert payload["meta"]["scene_fingerprint"] == scene["fingerprint"]

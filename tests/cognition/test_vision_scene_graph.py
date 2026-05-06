from __future__ import annotations

from eibrain.cognition.vision_scene_graph import (
    build_scene_graph,
    build_vision_scene_graph,
    to_eiprotocol_scene_content,
)


def _det(label: str, bbox: tuple[float, float, float, float], confidence: float = 0.9) -> dict[str, object]:
    x_min, y_min, x_max, y_max = bbox
    return {
        "label": label,
        "confidence": confidence,
        "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
    }


def test_builds_spatial_regions_people_objects_and_depth_from_detections_and_tracks() -> None:
    scene = build_vision_scene_graph(
        detections=[
            _det("cup", (0.43, 0.38, 0.51, 0.52), 0.82),
            _det("monitor", (0.72, 0.20, 0.92, 0.48), 0.88),
        ],
        tracks=[
            {
                "track_id": "person-007",
                "label": "person",
                "confidence": 0.94,
                "bbox": {"x_min": 0.05, "y_min": 0.20, "x_max": 0.32, "y_max": 0.86},
            }
        ],
        frame_metadata={
            "frame_id": "frame-001",
            "observed_at": "2026-05-06T09:00:00.000+08:00",
            "width": 1280,
            "height": 720,
        },
    )

    assert scene["frameId"] == "frame-001"
    assert scene["observedAt"] == "2026-05-06T09:00:00.000+08:00"
    assert [person["id"] for person in scene["people"]] == ["person-007"]
    assert scene["people"][0]["looking_at_device"] is None
    assert {item["label"] for item in scene["objects"]} == {"person", "cup", "monitor"}
    assert {item["label"] for item in scene["regions"]["left"]} == {"person"}
    assert {item["label"] for item in scene["regions"]["center"]} == {"cup"}
    assert {item["label"] for item in scene["regions"]["right"]} == {"monitor"}
    assert scene["people"][0]["depth"] == "near"
    assert next(item for item in scene["objects"] if item["label"] == "cup")["depth"] == "far"
    assert scene["metadata"]["width"] == 1280
    assert scene["metadata"]["height"] == 720


def test_builds_overlap_near_and_directional_relations() -> None:
    scene = build_vision_scene_graph(
        detections=[
            _det("person", (0.10, 0.20, 0.36, 0.82), 0.91),
            _det("phone", (0.27, 0.55, 0.39, 0.70), 0.84),
            _det("book", (0.42, 0.56, 0.52, 0.70), 0.77),
        ],
        frame_metadata={"frame_id": "frame-rel"},
    )

    relation_types = {
        (item["subjectLabel"], item["relation"], item["objectLabel"])
        for item in scene["relations"]
    }

    assert ("person", "overlaps", "phone") in relation_types
    assert ("person", "near", "book") in relation_types
    assert ("person", "left_of", "book") in relation_types
    assert scene["relationships"] == scene["relations"]


def test_selects_dominant_target_by_salience_and_summarizes_scene() -> None:
    scene = build_vision_scene_graph(
        detections=[
            _det("cup", (0.08, 0.42, 0.20, 0.62), 0.99),
            _det("person", (0.36, 0.16, 0.63, 0.88), 0.9),
            _det("chair", (0.70, 0.54, 0.84, 0.74), 0.8),
        ],
        frame_metadata={"frame_id": "frame-dom"},
    )

    assert scene["dominant_target"]["label"] == "person"
    assert scene["dominant_target"]["id"].startswith("person:")
    assert scene["dominant_target"]["region"] == "center_middle"
    assert "Observed chair, cup, person" in scene["summary"]
    assert "dominant target person in center_middle" in scene["summary"]


def test_empty_scene_keeps_stable_shape_and_safety_placeholders() -> None:
    scene = build_vision_scene_graph(
        detections=[],
        tracks=[],
        frame_metadata={"frame_id": "frame-empty", "observed_at": "2026-05-06T09:00:01.000+08:00"},
    )

    assert scene["people"] == []
    assert scene["objects"] == []
    assert scene["relations"] == []
    assert scene["regions"] == {"left": [], "center": [], "right": [], "near": [], "far": []}
    assert scene["dominant_target"] == {}
    assert scene["safety"] == {"pathClear": None, "nearObstacle": None}
    assert scene["summary"] == "Observed empty visual scene; pathClear unknown; nearObstacle unknown"


def test_accepts_payload_style_input_and_maps_to_eiprotocol_content() -> None:
    scene = build_scene_graph(
        {
            "frame_id": "frame-payload",
            "observed_at": "2026-05-06T09:00:02.000+08:00",
            "image_url": "memory://frame-payload.jpg",
            "detections": [_det("bottle", (0.68, 0.30, 0.76, 0.55), 0.86)],
        }
    )

    content = to_eiprotocol_scene_content(scene)

    assert scene["frameId"] == "frame-payload"
    assert scene["metadata"]["imageUrl"] == "memory://frame-payload.jpg"
    assert content["sceneId"] == scene["sceneId"]
    assert content["observedAt"] == "2026-05-06T09:00:02.000+08:00"
    assert content["objects"] == scene["objects"]
    assert content["relationships"] == scene["relations"]
    assert content["metadata"]["frameId"] == "frame-payload"

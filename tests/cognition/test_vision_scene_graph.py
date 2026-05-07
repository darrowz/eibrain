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


def test_normalizes_multimodal_features_provenance_and_lightweight_events() -> None:
    scene = build_vision_scene_graph(
        detections=[
            {
                "track_id": "person-001",
                "label": "person",
                "confidence": 0.93,
                "bbox": {"x_min": 0.20, "y_min": 0.18, "x_max": 0.58, "y_max": 0.92},
                "pose": {
                    "keypoints": [
                        {"name": "nose", "x": 0.40, "y": 0.27, "confidence": 0.88},
                        {"name": "right_wrist", "x": 0.61, "y": 0.58, "confidence": 0.76},
                    ]
                },
                "clip_labels": ["person", {"label": "operator", "confidence": 0.62}],
                "semantic_labels": ["human", "foreground"],
                "depth_m": 0.72,
                "distance_band": "near",
                "looking_at_device": True,
                "source": "hailo",
                "model_id": "yolov8n-pose+clip-placeholder",
                "provenance": {"device": "hailo8l", "pipeline": "pose_clip_depth"},
            },
            {
                "track_id": "phone-001",
                "label": "phone",
                "confidence": 0.84,
                "bbox": {"x_min": 0.58, "y_min": 0.52, "x_max": 0.68, "y_max": 0.66},
                "clip_labels": [{"label": "smartphone", "confidence": 0.71, "source": "cpu_clip"}],
                "semantic_labels": ["device"],
                "depth_m": 0.78,
                "source": "cpu",
                "model_id": "depth-placeholder",
            },
            {
                "track_id": "cup-001",
                "label": "cup",
                "confidence": 0.81,
                "bbox": {"x_min": 0.72, "y_min": 0.54, "x_max": 0.80, "y_max": 0.68},
                "depth_m": 2.4,
            },
        ],
        frame_metadata={"frame_id": "frame-multimodal", "source": "hailo"},
    )

    person = next(item for item in scene["objects"] if item["id"] == "person-001")
    phone = next(item for item in scene["objects"] if item["id"] == "phone-001")
    cup = next(item for item in scene["objects"] if item["id"] == "cup-001")
    event_types = {event["eventType"] for event in scene["events"]}
    relation_types = {
        (item["subjectId"], item["relation"], item["objectId"])
        for item in scene["relations"]
    }

    assert person["pose"]["keypoints"][1] == {"name": "right_wrist", "x": 0.61, "y": 0.58, "confidence": 0.76}
    assert person["clip_labels"] == [
        {"label": "person"},
        {"label": "operator", "confidence": 0.62},
    ]
    assert person["semantic_labels"] == ["human", "foreground"]
    assert person["depth_m"] == 0.72
    assert person["distance_band"] == "near"
    assert person["depth"] == "near"
    assert person["source"] == "hailo"
    assert person["model_id"] == "yolov8n-pose+clip-placeholder"
    assert person["provenance"]["device"] == "hailo8l"
    assert phone["clip_labels"] == [{"label": "smartphone", "confidence": 0.71, "source": "cpu_clip"}]
    assert phone["provenance"]["source"] == "cpu"
    assert cup["distance_band"] == "far"
    assert "looking_at_device" in event_types
    assert "hand_near_object" in event_types
    assert ("person-001", "hand_near_object", "phone-001") in relation_types
    assert scene["event_summary"]


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


def test_scene_graph_accepts_protocol_bbox_depth_distance_and_exports_multimodal_fields() -> None:
    scene = build_scene_graph(
        {
            "frame_id": "frame-protocol",
            "observed_at": "2026-05-06T09:00:03.000+08:00",
            "detections": [
                {
                    "trackId": "person-protocol-1",
                    "label": "person",
                    "confidence": 0.93,
                    "bbox": {"x": 0.20, "y": 0.18, "w": 0.38, "h": 0.74},
                    "clipLabels": [{"label": "person at desk", "score": 0.84}],
                    "semanticLabels": [{"label": "workspace", "confidence": 0.79}],
                    "depth": {"median": 0.72, "unit": "m"},
                    "distance": {"fromCameraM": 0.72},
                    "pose": {"keypoints": [{"name": "nose", "x": 0.40, "y": 0.27, "score": 0.88}]},
                    "source": "hailo",
                    "modelId": "yolov8n-pose",
                }
            ],
        }
    )

    obj = scene["objects"][0]
    content = to_eiprotocol_scene_content(scene)

    assert obj["bbox"] == {"x_min": 0.2, "y_min": 0.18, "x_max": 0.58, "y_max": 0.92}
    assert obj["depth_m"] == 0.72
    assert obj["distance_band"] == "near"
    assert obj["clip_labels"][0]["label"] == "person at desk"
    assert content["clipLabels"][0]["label"] == "person at desk"
    assert content["semanticLabels"][0]["label"] == "workspace"
    assert content["depth"]["median"] == 0.72
    assert content["distance"]["fromCameraM"] == 0.72
    assert content["sceneGraph"]["nodes"][0]["id"] == "person-protocol-1"
    assert content["sceneGraphProvenance"]["builder"] == "vision_scene_graph"


def test_scene_graph_accepts_protocol_list_bbox_and_preserves_tracking_diagnostics() -> None:
    scene = build_scene_graph(
        {
            "frame_id": "frame-list-protocol",
            "detections": [
                {
                    "trackId": "person-list-1",
                    "label": "person",
                    "score": 0.91,
                    "bbox": [0.62, 0.35, 0.12, 0.18],
                    "trackingDiagnostics": {
                        "trackIdSwitchCount": 0,
                        "targetStabilityRatio": 1.0,
                    },
                }
            ],
        }
    )

    obj = scene["objects"][0]
    content = to_eiprotocol_scene_content(scene)

    assert obj["bbox"] == {"x_min": 0.62, "y_min": 0.35, "x_max": 0.74, "y_max": 0.53}
    assert obj["tracking_diagnostics"]["trackIdSwitchCount"] == 0
    assert content["trackingDiagnostics"]["targetStabilityRatio"] == 1.0


def test_scene_graph_honors_explicit_normalized_list_xyxy_bbox() -> None:
    scene = build_scene_graph(
        {
            "frame_id": "frame-list-xyxy",
            "detections": [
                {
                    "trackId": "person-list-xyxy",
                    "label": "person",
                    "score": 0.91,
                    "bbox": [0.62, 0.35, 0.74, 0.53],
                    "bboxFormat": "xyxy",
                }
            ],
        }
    )

    assert scene["objects"][0]["bbox"] == {"x_min": 0.62, "y_min": 0.35, "x_max": 0.74, "y_max": 0.53}
    assert scene["objects"][0]["center"] == {"x": 0.68, "y": 0.44}


def test_scene_graph_merges_track_and_detection_with_same_track_id() -> None:
    scene = build_vision_scene_graph(
        tracks=[
            {
                "trackId": "person-merged",
                "label": "person",
                "confidence": 0.88,
                "bbox": {"x_min": 0.20, "y_min": 0.18, "x_max": 0.58, "y_max": 0.92},
            }
        ],
        detections=[
            {
                "trackId": "person-merged",
                "label": "person",
                "confidence": 0.93,
                "bbox": {"x_min": 0.20, "y_min": 0.18, "x_max": 0.58, "y_max": 0.92},
                "clipLabels": [{"label": "person at desk", "score": 0.84}],
            }
        ],
        frame_metadata={"frame_id": "frame-merge"},
    )

    assert len(scene["objects"]) == 1
    assert scene["objects"][0]["id"] == "person-merged"
    assert scene["objects"][0]["clip_labels"][0]["label"] == "person at desk"


def test_scene_graph_derives_temporal_states_and_event_summary_from_previous_scene() -> None:
    previous = build_vision_scene_graph(
        tracks=[
            {
                "track_id": "person-007",
                "label": "person",
                "confidence": 0.9,
                "bbox": {"x_min": 0.40, "y_min": 0.25, "x_max": 0.58, "y_max": 0.75},
            },
            {
                "track_id": "cup-001",
                "label": "cup",
                "confidence": 0.82,
                "bbox": {"x_min": 0.12, "y_min": 0.50, "x_max": 0.22, "y_max": 0.62},
            },
            {
                "track_id": "book-001",
                "label": "book",
                "confidence": 0.74,
                "bbox": {"x_min": 0.72, "y_min": 0.52, "x_max": 0.84, "y_max": 0.64},
            },
        ],
        frame_metadata={"frame_id": "frame-prev"},
    )

    current = build_vision_scene_graph(
        tracks=[
            {
                "track_id": "person-007",
                "label": "person",
                "confidence": 0.92,
                "bbox": {"x_min": 0.34, "y_min": 0.18, "x_max": 0.64, "y_max": 0.88},
            },
            {
                "track_id": "cup-001",
                "label": "cup",
                "confidence": 0.83,
                "bbox": {"x_min": 0.121, "y_min": 0.501, "x_max": 0.221, "y_max": 0.621},
            },
            {
                "track_id": "plant-001",
                "label": "plant",
                "confidence": 0.7,
                "bbox": {"x_min": 0.78, "y_min": 0.20, "x_max": 0.92, "y_max": 0.48},
            },
        ],
        frame_metadata={"frame_id": "frame-current"},
        previous_scene=previous,
        motion_threshold=0.08,
        approach_area_delta=0.04,
    )

    states = {item["trackId"]: item["temporalState"] for item in current["objects"]}
    temporal_events = {(event["eventType"], event["trackId"]) for event in current["temporal"]["events"]}

    assert states["person-007"] == "approaching"
    assert states["cup-001"] == "stationary"
    assert states["plant-001"] == "appeared"
    assert ("disappeared", "book-001") in temporal_events
    assert ("approaching", "person-007") in temporal_events
    assert "stationary cup-001" in current["temporal"]["eventSummary"]
    assert "disappeared book-001" in current["event_summary"]

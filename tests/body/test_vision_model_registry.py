from __future__ import annotations


def test_registry_declares_supported_capabilities_and_backends() -> None:
    from eibrain.body.vision_model_registry import SUPPORTED_BACKENDS
    from eibrain.body.vision_model_registry import SUPPORTED_CAPABILITIES
    from eibrain.body.vision_model_registry import available_profiles

    assert SUPPORTED_CAPABILITIES == frozenset(
        {"detection", "face", "pose", "segmentation", "depth", "clip_scene", "tracking"}
    )
    assert SUPPORTED_BACKENDS == frozenset({"hailo", "rpicam", "gstreamer", "opencv"})
    assert {profile.backend for profile in available_profiles()} == SUPPORTED_BACKENDS


def test_selects_default_yolov8s_h8_for_hailo8_detection() -> None:
    from eibrain.body.vision_model_registry import select_profile

    selection = select_profile(
        device_capabilities={"hailo8"},
        target_fps=15.0,
        required_capabilities={"detection", "tracking"},
    )

    assert selection.profile is not None
    assert selection.profile.model_id == "yolov8s_h8"
    assert selection.profile.backend == "gstreamer"
    assert selection.profile.capabilities == frozenset({"detection", "tracking"})
    assert selection.missing_capabilities == frozenset()
    assert selection.diagnostics["status"] == "ok"


def test_hailo8l_downgrades_to_smaller_detection_profile() -> None:
    from eibrain.body.vision_model_registry import select_profile

    selection = select_profile(
        device_capabilities={"hailo8l"},
        target_fps=10.0,
        required_capabilities={"detection", "tracking"},
    )

    assert selection.profile is not None
    assert selection.profile.model_id == "yolov8n_h8l"
    assert selection.profile.device == "hailo8l"
    assert selection.profile.degraded_from == "yolov8s_h8"
    assert selection.diagnostics["degraded_from"] == "yolov8s_h8"


def test_cpu_fallback_selects_opencv_detection_without_loading_model() -> None:
    from eibrain.body.vision_model_registry import select_profile

    selection = select_profile(
        device_capabilities={"cpu"},
        target_fps=4.0,
        required_capabilities={"detection"},
    )

    assert selection.profile is not None
    assert selection.profile.model_id == "opencv_cpu_detector"
    assert selection.profile.backend == "opencv"
    assert selection.profile.loadable is False
    assert selection.diagnostics["status"] == "ok"


def test_requirement_matching_prefers_profile_covering_all_required_capabilities() -> None:
    from eibrain.body.vision_model_registry import select_profile

    selection = select_profile(
        device_capabilities={"hailo8l"},
        target_fps=8.0,
        required_capabilities={"detection", "face", "tracking"},
    )

    assert selection.profile is not None
    assert selection.profile.model_id == "personface_h8l"
    assert selection.profile.supports({"detection", "face"})
    assert selection.missing_capabilities == frozenset()


def test_missing_capability_diagnostics_explain_unmet_requirements() -> None:
    from eibrain.body.vision_model_registry import select_profile

    selection = select_profile(
        device_capabilities={"hailo8"},
        target_fps=12.0,
        required_capabilities={"detection", "depth"},
        allowed_backends={"gstreamer", "hailo"},
    )

    assert selection.profile is None
    assert selection.missing_capabilities == frozenset({"depth"})
    assert selection.diagnostics["status"] == "missing_capabilities"
    assert selection.diagnostics["missing_capabilities"] == ["depth"]
    assert "depth" in selection.reason
    assert "gstreamer" in selection.reason

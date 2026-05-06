from apps.body_runtime.vision_soak import summarize_vision_soak


def test_vision_soak_marks_healthy_stream_as_passing():
    samples = [
        {
            "fps": 9.8,
            "target_fps": 10.0,
            "frame_age_ms": 110.0,
            "loop_elapsed_ms": 78.0,
            "dropped_frames": 0,
            "service_state": "ok",
        },
        {
            "fps": 10.2,
            "target_fps": 10.0,
            "frame_age_ms": 130.0,
            "loop_elapsed_ms": 82.0,
            "dropped_frames": 0,
            "service_state": "ok",
        },
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 125.0,
            "loop_elapsed_ms": 80.0,
            "dropped_frames": 0,
            "service_state": "ok",
        },
    ]

    summary = summarize_vision_soak(samples)

    assert summary["pass"] is True
    assert summary["sample_count"] == 3
    assert summary["fps"]["avg"] == 10.0
    assert summary["fps_ratio"] == 1.0
    assert summary["service_ok_ratio"] == 1.0
    assert summary["bottleneck_reason"] == "healthy"


def test_vision_soak_detects_low_fps_bottleneck():
    samples = [
        {"fps": 4.5, "target_fps": 10.0, "frame_age_ms": 120.0, "service_state": "ok"},
        {"fps": 5.0, "target_fps": 10.0, "frame_age_ms": 130.0, "service_state": "ok"},
        {"fps": 5.5, "target_fps": 10.0, "frame_age_ms": 140.0, "service_state": "ok"},
    ]

    summary = summarize_vision_soak(samples)

    assert summary["pass"] is False
    assert summary["fps_ratio"] == 0.5
    assert summary["bottleneck_reason"] == "low_fps"


def test_vision_soak_detects_stale_frames_by_p95_age():
    samples = [
        {"fps": 10.0, "target_fps": 10.0, "frame_age_ms": 90.0, "service_state": "ok"},
        {"fps": 10.0, "target_fps": 10.0, "frame_age_ms": 120.0, "service_state": "ok"},
        {"fps": 10.0, "target_fps": 10.0, "frame_age_ms": 900.0, "service_state": "ok"},
    ]

    summary = summarize_vision_soak(samples, max_p95_frame_age_ms=500.0)

    assert summary["pass"] is False
    assert summary["frame_age_ms"]["p95"] >= 800.0
    assert summary["stale_ratio"] == 1 / 3
    assert summary["bottleneck_reason"] == "stale_frames"


def test_vision_soak_detects_service_instability_and_drops():
    samples = [
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 90.0,
            "dropped_frames": 0,
            "service_state": "ok",
        },
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 90.0,
            "dropped_frames": 2,
            "service_state": "error",
        },
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 90.0,
            "dropped_frames": 3,
            "service_state": "ok",
        },
    ]

    summary = summarize_vision_soak(samples, max_drop_rate=0.2)

    assert summary["pass"] is False
    assert summary["service_ok_ratio"] == 2 / 3
    assert summary["drop_rate"] == 5 / 3
    assert summary["bottleneck_reason"] == "service_unstable"


def test_vision_soak_handles_empty_samples():
    summary = summarize_vision_soak([])

    assert summary["pass"] is False
    assert summary["sample_count"] == 0
    assert summary["bottleneck_reason"] == "no_samples"

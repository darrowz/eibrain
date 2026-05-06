import json

from apps.body_runtime.vision_soak import collect_vision_soak
from apps.body_runtime.vision_soak import normalize_vision_status_sample
from apps.body_runtime.vision_soak import run_synthetic_vision_soak
from apps.body_runtime.vision_soak import run_vision_soak
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
    assert summary["drop_rate"] == 1.0
    assert summary["bottleneck_reason"] == "service_unstable"


def test_vision_soak_handles_empty_samples():
    summary = summarize_vision_soak([])

    assert summary["pass"] is False
    assert summary["sample_count"] == 0
    assert summary["bottleneck_reason"] == "no_samples"


def test_vision_soak_reports_tracking_diagnostics_from_trace_samples():
    samples = [
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 80.0,
            "service_state": "ok",
            "event_count": 1,
            "dropout": False,
            "stable_target_track_id": "person-001",
            "tracks": [{"synthetic_subject_id": "primary", "track_id": "person-001"}],
        },
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 90.0,
            "service_state": "ok",
            "event_count": 0,
            "dropout": True,
            "stable_target_track_id": "person-001",
            "tracks": [],
        },
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 95.0,
            "service_state": "ok",
            "event_count": 0,
            "dropout": False,
            "stable_target_track_id": "person-001",
            "tracks": [{"synthetic_subject_id": "primary", "track_id": "person-001"}],
        },
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 100.0,
            "service_state": "ok",
            "event_count": 1,
            "dropout": False,
            "stable_target_track_id": "person-002",
            "tracks": [{"synthetic_subject_id": "primary", "track_id": "person-002"}],
        },
    ]

    summary = summarize_vision_soak(samples)

    assert summary["track_id_switch_count"] == 1
    assert summary["target_stability_ratio"] == 2 / 3
    assert summary["event_rate_hz"] == 5.0
    assert summary["frame_drop_tolerance"] == 1
    assert summary["p95_frame_age_ms"] == 100.0
    assert summary["metadata"]["trace"]["metrics"]["track_id_switch_count"] == 1
    assert summary["metadata"]["web"]["kind"] == "vision_soak_summary"


def test_synthetic_vision_soak_covers_long_tracking_stressors():
    summary = run_synthetic_vision_soak(frame_count=36, target_fps=10.0)

    assert summary["pass"] is True
    assert summary["sample_count"] == 36
    assert summary["track_id_switch_count"] >= 1
    assert summary["target_stability_ratio"] >= 0.75
    assert summary["event_rate_hz"] > 0.0
    assert summary["frame_drop_tolerance"] >= 1
    assert summary["p95_frame_age_ms"] > 0.0
    assert summary["scenario_coverage"] == {
        "jitter": True,
        "dropout": True,
        "track_id_switch": True,
        "target_swap": True,
        "short_loss_recovery": True,
    }
    assert summary["metadata"]["hailo"]["backend"] == "synthetic_hailo"
    assert summary["metadata"]["trace"]["name"] == "vision_soak.synthetic_long_tracking"


def test_collect_vision_soak_normalizes_fake_status_samples_without_hardware():
    clock = _FakeClock()
    samples = iter(
        [
            {
                "visual_diagnostics": {
                    "vision_fps": 9.5,
                    "vision_target_fps": 10.0,
                    "vision_frame_age_s": 0.12,
                    "status": "ok",
                    "soak_summary": {
                        "track_id_switch_count": 0,
                        "target_stability_ratio": 1.0,
                        "event_rate_hz": 0.5,
                        "frame_drop_tolerance": 0,
                    },
                }
            },
            {
                "eye": {
                    "diagnostics": {
                        "fps": 10.5,
                        "frame_age_ms": 140.0,
                        "event_count": 1,
                        "stable_target": {"track_id": "person-001"},
                    },
                    "status": "ok",
                    "tracks": [{"subject_id": "primary", "track_id": "person-001"}],
                }
            },
        ]
    )

    summary = collect_vision_soak(
        lambda: next(samples),
        duration_s=2.0,
        interval_s=1.0,
        target_fps=10.0,
        clock=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert summary["pass"] is True
    assert summary["sample_count"] == 2
    assert summary["fps"]["avg"] == 10.0
    assert summary["p95_frame_age_ms"] == 140.0
    assert summary["service_ok_ratio"] == 1.0
    assert summary["collection"]["error_count"] == 0
    assert summary["collection"]["duration_s"] == 2.0


def test_vision_soak_normalizes_operator_console_visual_diagnostics_shape():
    sample = normalize_vision_status_sample(
        {
            "visual_diagnostics": {
                "vision_fps": 9.7,
                "vision_target_fps": 10.0,
                "vision_frame_age_s": 0.08,
                "vision_service_status": "ok",
                "data_status": "live",
                "tracking_status": "tracking",
                "tracking_target": {
                    "trackId": "person-7",
                    "label": "face",
                    "target_x": 0.62,
                    "score": 0.91,
                },
                "soak_summary": {
                    "track_id_switch_count": 0,
                    "target_stability_ratio": 1.0,
                    "event_rate_hz": 0.2,
                    "frame_drop_tolerance": 0,
                },
            }
        },
        elapsed_s=1.0,
    )

    assert sample["service_state"] == "ok"
    assert sample["fps"] == 9.7
    assert sample["frame_age_ms"] == 80.0
    assert sample["stable_target"]["trackId"] == "person-7"
    assert sample["tracks"][0]["trackId"] == "person-7"


def test_target_stability_threshold_ignores_missing_target_when_only_zero_placeholder_exists():
    samples = [
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 90.0,
            "service_state": "ok",
            "target_stability_ratio": 0.0,
        },
        {
            "fps": 10.0,
            "target_fps": 10.0,
            "frame_age_ms": 95.0,
            "service_state": "ok",
            "target_stability_ratio": 0.0,
        },
    ]

    summary = summarize_vision_soak(samples, min_target_stability_ratio=0.5)

    assert summary["pass"] is True
    assert summary["target_stability_ratio"] == 1.0


def test_drop_rate_treats_monotonic_dropped_frames_as_cumulative_counter():
    samples = [
        {"fps": 10.0, "target_fps": 10.0, "frame_age_ms": 90.0, "service_state": "ok", "dropped_frames": 0},
        {"fps": 10.0, "target_fps": 10.0, "frame_age_ms": 90.0, "service_state": "ok", "dropped_frames": 1},
        {"fps": 10.0, "target_fps": 10.0, "frame_age_ms": 90.0, "service_state": "ok", "dropped_frames": 1},
        {"fps": 10.0, "target_fps": 10.0, "frame_age_ms": 90.0, "service_state": "ok", "dropped_frames": 1},
    ]

    summary = summarize_vision_soak(samples, max_drop_rate=0.3)

    assert summary["drop_rate"] == 0.25
    assert summary["pass"] is True


def test_collect_vision_soak_records_status_source_errors():
    clock = _FakeClock()

    def source():
        if clock.now < 1.0:
            return {"diagnostics": {"fps": 10.0, "frame_age_ms": 100.0}, "status": "ok"}
        raise OSError("status endpoint unavailable")

    summary = collect_vision_soak(
        source,
        duration_s=2.0,
        interval_s=1.0,
        target_fps=10.0,
        clock=clock.monotonic,
        sleeper=clock.sleep,
    )

    assert summary["pass"] is False
    assert summary["sample_count"] == 2
    assert summary["service_ok_ratio"] == 0.5
    assert summary["bottleneck_reason"] == "service_unstable"
    assert summary["collection"]["error_count"] == 1
    assert "status endpoint unavailable" in summary["collection"]["errors"][0]["message"]


def test_collect_vision_soak_handles_empty_collection_window():
    summary = collect_vision_soak(lambda: {"status": "ok"}, duration_s=0.0, interval_s=1.0)

    assert summary["pass"] is False
    assert summary["sample_count"] == 0
    assert summary["bottleneck_reason"] == "no_samples"
    assert summary["collection"]["requested_duration_s"] == 0.0


def test_run_vision_soak_writes_json_summary(tmp_path):
    output_path = tmp_path / "vision-soak-summary.json"

    summary = run_vision_soak(
        status_source=lambda: {"diagnostics": {"fps": 10.0, "frame_age_ms": 100.0}, "status": "ok"},
        duration_s=1.0,
        interval_s=1.0,
        output_path=output_path,
        target_fps=10.0,
        clock=_StaticClock().monotonic,
        sleeper=lambda _: None,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["sample_count"] == 1
    assert payload["fps"]["avg"] == 10.0
    assert payload == summary


def test_vision_soak_cli_passes_runtime_arguments(monkeypatch, tmp_path):
    from apps.body_runtime import vision_soak_cli

    output_path = tmp_path / "summary.json"
    captured = {}

    def fake_run_vision_soak(**kwargs):
        captured.update(kwargs)
        return {"pass": True, "sample_count": 1}

    monkeypatch.setattr(vision_soak_cli, "run_vision_soak", fake_run_vision_soak)

    exit_code = vision_soak_cli.main(
        [
            "--duration",
            "30",
            "--interval",
            "2",
            "--status-url",
            "http://honjia.local:18080/status.json",
            "--output-path",
            str(output_path),
            "--target-fps",
            "12",
            "--min-fps-ratio",
            "0.7",
            "--max-p95-frame-age-ms",
            "650",
            "--max-drop-rate",
            "0.2",
            "--min-service-ok-ratio",
            "0.9",
        ]
    )

    assert exit_code == 0
    assert captured["duration_s"] == 30.0
    assert captured["interval_s"] == 2.0
    assert captured["status_url"] == "http://honjia.local:18080/status.json"
    assert captured["output_path"] == output_path
    assert captured["target_fps"] == 12.0
    assert captured["thresholds"]["min_fps_ratio"] == 0.7
    assert captured["thresholds"]["max_p95_frame_age_ms"] == 650.0
    assert captured["thresholds"]["max_drop_rate"] == 0.2
    assert captured["thresholds"]["min_service_ok_ratio"] == 0.9


def test_vision_soak_cli_returns_nonzero_for_failing_summary(monkeypatch):
    from apps.body_runtime import vision_soak_cli

    monkeypatch.setattr(vision_soak_cli, "run_vision_soak", lambda **kwargs: {"pass": False, "fail_reason": "low_fps"})

    assert vision_soak_cli.main(["--duration", "1", "--interval", "1"]) == 1


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _StaticClock:
    def __init__(self) -> None:
        self.calls = 0

    def monotonic(self) -> float:
        self.calls += 1
        return 0.0 if self.calls == 1 else 1.0

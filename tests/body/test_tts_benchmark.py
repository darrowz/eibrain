from __future__ import annotations


def test_tts_benchmark_collects_backend_metrics() -> None:
    from apps.body_runtime.benchmark_tts import benchmark_backend

    calls = []
    samples = iter([1.0, 1.25, 2.0, 2.75])

    def _speak_fn(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "details": {"generate_elapsed_ms": 123.0}}

    report = benchmark_backend(
        backend="moss_onnx",
        texts=["first", "second"],
        output_device="plughw:2,0",
        speak_fn=_speak_fn,
        perf_counter=lambda: next(samples),
        model="/models/moss",
        voice_id="Junhao",
    )

    assert report["backend"] == "moss_onnx"
    assert report["summary"]["count"] == 2
    assert report["summary"]["ok_count"] == 2
    assert report["summary"]["avg_elapsed_ms"] == 500.0
    assert report["items"][0]["elapsed_ms"] == 250.0
    assert report["items"][1]["elapsed_ms"] == 750.0
    assert calls[0]["text"] == "first"
    assert calls[0]["backend"] == "moss_onnx"
    assert calls[0]["model"] == "/models/moss"
    assert calls[0]["voice_id"] == "Junhao"

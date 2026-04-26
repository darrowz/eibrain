"""Standalone Hailo vision service for honjia.

The service owns /dev/video0 and /dev/hailo0, then publishes a small
state.json contract consumed by EyeOrgan. This keeps monitoring and tracking
from synchronously grabbing frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import signal
import time
from typing import Any

from eibrain.body.runtime_linux import capture_frame, run_hailo_frame_inference
from eibrain.body.vision_state import DEFAULT_VISION_FRAME_PATH
from eibrain.body.vision_state import DEFAULT_VISION_STATE_PATH
from eibrain.body.vision_state import VisionStateWriter
from eibrain.body.vision_state import build_vision_state
from eibrain.infra.config import EIBrainConfig, load_config


class SingleFrameHailoDetector:
    """Compatibility detector used as a fallback when GStreamer is unavailable."""

    def __init__(
        self,
        *,
        camera_device: str,
        frame_path: str | Path,
        hef_path: str,
        labels: list[str],
        score_threshold: float,
        input_format: str = "mjpeg",
        video_size: str = "640x480",
        timeout_s: float = 5.0,
    ) -> None:
        self.camera_device = camera_device
        self.frame_path = Path(frame_path)
        self.hef_path = hef_path
        self.labels = labels
        self.score_threshold = score_threshold
        self.input_format = input_format
        self.video_size = video_size
        self.timeout_s = timeout_s

    def detect_once(self) -> dict[str, Any]:
        captured_at = time.time()
        capture_result = capture_frame(
            device=self.camera_device,
            output_path=self.frame_path,
            input_format=self.input_format,
            video_size=self.video_size,
            timeout_s=self.timeout_s,
        )
        if capture_result.get("status") != "ok":
            return build_vision_state(
                detections=[],
                frame_path=self.frame_path,
                status="capture_failed",
                frame_captured_at_ts=captured_at,
                backend="hailort_single_frame",
                details={"capture_result": capture_result},
            )
        inference = run_hailo_frame_inference(
            image_path=self.frame_path,
            hef_path=self.hef_path,
            labels=self.labels,
            score_threshold=self.score_threshold,
        )
        details = dict(inference.get("details", {})) if isinstance(inference.get("details"), dict) else {}
        detections = details.get("detections", [])
        if not isinstance(detections, list):
            detections = []
        return build_vision_state(
            detections=[item for item in detections if isinstance(item, dict)],
            frame_path=self.frame_path,
            status="ok" if inference.get("status") == "ok" else "inference_failed",
            frame_captured_at_ts=captured_at,
            backend="hailort_single_frame",
            details={"capture_result": capture_result, "inference_details": details},
        )


class GStreamerHailoDetector:
    """GStreamer/Hailo detector that reads Hailo metadata from appsink."""

    def __init__(
        self,
        *,
        camera_device: str,
        frame_path: str | Path,
        hef_path: str,
        postprocess_so_path: str,
        postprocess_config_path: str,
        postprocess_function: str,
        score_threshold: float,
        width: int = 640,
        height: int = 480,
        framerate: int = 30,
    ) -> None:
        self.camera_device = camera_device
        self.frame_path = Path(frame_path)
        self.hef_path = hef_path
        self.postprocess_so_path = postprocess_so_path
        self.postprocess_config_path = postprocess_config_path
        self.postprocess_function = postprocess_function
        self.score_threshold = score_threshold
        self.width = width
        self.height = height
        self.framerate = framerate
        self._pipeline = None
        self._appsink = None
        self._latest_state: dict[str, Any] | None = None

    def start(self) -> None:
        gi, Gst, _GLib, _hailo = _load_gstreamer_modules()
        gi.require_version("Gst", "1.0")
        Gst.init(None)
        self.frame_path.parent.mkdir(parents=True, exist_ok=True)
        pipeline_text = self._pipeline_text()
        self._pipeline = Gst.parse_launch(pipeline_text)
        self._appsink = self._pipeline.get_by_name("metadata_sink")
        self._appsink.connect("new-sample", self._on_sample)
        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("failed to start Hailo GStreamer pipeline")

    def stop(self) -> None:
        if self._pipeline is None:
            return
        _gi, Gst, _GLib, _hailo = _load_gstreamer_modules()
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None

    def detect_once(self) -> dict[str, Any]:
        if self._pipeline is None:
            self.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._latest_state is not None:
                state = dict(self._latest_state)
                self._latest_state = None
                return state
            time.sleep(0.02)
        return build_vision_state(
            detections=[],
            frame_path=self.frame_path,
            status="no_metadata",
            backend="gstreamer_hailo",
            details={"pipeline": self._pipeline_text()},
        )

    def _on_sample(self, appsink) -> object:
        _gi, Gst, _GLib, hailo = _load_gstreamer_modules()
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        detections: list[dict[str, Any]] = []
        try:
            roi = hailo.get_roi_from_buffer(buffer)
            for detection in roi.get_objects_typed(hailo.HAILO_DETECTION):
                confidence = float(detection.get_confidence())
                if confidence < self.score_threshold:
                    continue
                bbox = detection.get_bbox()
                item: dict[str, Any] = {
                    "label": str(detection.get_label()),
                    "class_id": int(detection.get_class_id()),
                    "score": confidence,
                    "bbox": {
                        "x_min": float(bbox.xmin()),
                        "y_min": float(bbox.ymin()),
                        "x_max": float(bbox.xmax()),
                        "y_max": float(bbox.ymax()),
                    },
                }
                track_id = _read_track_id(detection)
                if track_id is not None:
                    item["track_id"] = track_id
                detections.append(item)
        except Exception as exc:
            self._latest_state = build_vision_state(
                detections=[],
                frame_path=self.frame_path,
                status="metadata_parse_failed",
                backend="gstreamer_hailo",
                details={"error": str(exc)},
            )
            return Gst.FlowReturn.OK
        self._latest_state = build_vision_state(
            detections=detections,
            frame_path=self.frame_path,
            status="ok",
            backend="gstreamer_hailo",
            details={"pipeline": "appsink_metadata"},
        )
        return Gst.FlowReturn.OK

    def _pipeline_text(self) -> str:
        frame_location = str(self.frame_path)
        return (
            f"v4l2src device={self.camera_device} io-mode=mmap ! "
            f"image/jpeg,width={self.width},height={self.height},framerate={self.framerate}/1 ! "
            "jpegdec ! videoconvert ! videoscale ! "
            "video/x-raw,format=RGB,width=640,height=640 ! "
            "tee name=t "
            "t. ! queue max-size-buffers=2 leaky=downstream ! "
            f"hailonet hef-path={self.hef_path} scheduling-algorithm=0 is-active=true force-writable=true ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            f"hailofilter so-path={self.postprocess_so_path} "
            f"config-path={self.postprocess_config_path} "
            f"function-name={self.postprocess_function} qos=false ! "
            "hailotracker class-id=-1 ! "
            "appsink name=metadata_sink emit-signals=true sync=false max-buffers=1 drop=true "
            "t. ! queue max-size-buffers=1 leaky=downstream ! "
            f"videoconvert ! jpegenc ! multifilesink location={frame_location} max-files=1"
        )


class VisionHailoService:
    def __init__(
        self,
        *,
        detector,
        writer: VisionStateWriter,
        interval_s: float = 0.2,
    ) -> None:
        self.detector = detector
        self.writer = writer
        self.interval_s = interval_s
        self._running = False

    def run_forever(self) -> None:
        self._running = True
        while self._running:
            started = time.monotonic()
            state = self.detector.detect_once()
            self.writer.write(state)
            sleep_s = max(0.0, self.interval_s - (time.monotonic() - started))
            if sleep_s:
                time.sleep(sleep_s)

    def stop(self) -> None:
        self._running = False
        stop = getattr(self.detector, "stop", None)
        if callable(stop):
            stop()


def detector_from_config(config: EIBrainConfig, *, backend: str) -> object:
    eye = config.body.organs.get("eye")
    camera = eye.subfunctions.get("camera") if eye is not None else None
    detection = eye.subfunctions.get("detection") if eye is not None else None
    camera_extra = camera.driver.extra if camera is not None else {}
    detection_extra = detection.driver.extra if detection is not None else {}
    frame_path = Path(str(detection_extra.get("frame_path", camera_extra.get("frame_path", DEFAULT_VISION_FRAME_PATH))))
    labels = _read_labels(detection_extra.get("labels"), default=["person", "face"])
    if backend == "single_frame":
        return SingleFrameHailoDetector(
            camera_device=str(camera_extra.get("device", "/dev/video0")),
            frame_path=frame_path,
            hef_path=str(detection_extra.get("hef_path", "/usr/share/hailo-models/yolov5s_personface_h8l.hef")),
            labels=labels,
            score_threshold=float(detection_extra.get("score_threshold", 0.3)),
            input_format=str(camera_extra.get("input_format", "mjpeg") or "mjpeg"),
            video_size=str(camera_extra.get("video_size", "640x480") or "640x480"),
            timeout_s=float(camera_extra.get("timeout_s", 5.0)),
        )
    return GStreamerHailoDetector(
        camera_device=str(camera_extra.get("device", "/dev/video0")),
        frame_path=frame_path,
        hef_path=str(detection_extra.get("hef_path", "/usr/share/hailo-models/yolov5s_personface_h8l.hef")),
        postprocess_so_path=str(
            detection_extra.get(
                "postprocess_so_path",
                "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes/libyolo_hailortpp_post.so",
            )
        ),
        postprocess_config_path=str(detection_extra.get("postprocess_config_path", "/usr/share/hailo-models/yolov5_personface.json")),
        postprocess_function=str(detection_extra.get("postprocess_function", "filter")),
        score_threshold=float(detection_extra.get("score_threshold", 0.3)),
        width=int(camera_extra.get("pipeline_width", 640)),
        height=int(camera_extra.get("pipeline_height", 480)),
        framerate=int(camera_extra.get("pipeline_framerate", 30)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eibrain Hailo vision service")
    parser.add_argument("--config", default="config/eibrain.yaml")
    parser.add_argument("--backend", choices=("gstreamer", "single_frame"), default="gstreamer")
    parser.add_argument("--state-path", default="")
    parser.add_argument("--interval-s", type=float, default=0.2)
    args = parser.parse_args()

    config = load_config(args.config)
    state_path = Path(args.state_path) if args.state_path else _state_path_from_config(config)
    detector = detector_from_config(config, backend=args.backend)
    service = VisionHailoService(detector=detector, writer=VisionStateWriter(state_path), interval_s=args.interval_s)

    def _stop(_signum, _frame) -> None:
        service.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(f"eibrain vision service writing {state_path} via {args.backend}", flush=True)
    service.run_forever()


def _state_path_from_config(config: EIBrainConfig) -> Path:
    eye = config.body.organs.get("eye")
    if eye is None:
        return DEFAULT_VISION_STATE_PATH
    for name in ("detection", "camera", "identity"):
        subfunction = eye.subfunctions.get(name)
        if subfunction is None:
            continue
        state_path = subfunction.driver.extra.get("state_path")
        if state_path:
            return Path(str(state_path))
    return DEFAULT_VISION_STATE_PATH


def _load_gstreamer_modules():
    import gi  # type: ignore

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import GLib, Gst  # type: ignore
    import hailo  # type: ignore

    return gi, Gst, GLib, hailo


def _read_track_id(detection) -> object | None:
    try:
        for unique_id in detection.get_objects_typed("HAILO_UNIQUE_ID"):
            return unique_id.get_id()
    except Exception:
        return None
    return None


def _read_labels(value: object, *, default: list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return list(default)


if __name__ == "__main__":
    if shutil.which("gst-launch-1.0") is None:
        pass
    main()

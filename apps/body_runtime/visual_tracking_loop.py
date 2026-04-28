"""Continuous visual tracking loop for honjia."""

from __future__ import annotations

import inspect
import threading
import time

from apps.body_runtime.app import BodyRuntimeApp
from apps.body_runtime.engagement_state import EngagementStateReader


class VisualTrackingLoop:
    def __init__(
        self,
        *,
        body_runtime: BodyRuntimeApp,
        interval_s: float = 0.5,
        recenter_after_misses: int = 8,
        source: str = "active",
        engagement_reader: EngagementStateReader | None = None,
        sleeping_interval_s: float = 2.0,
    ) -> None:
        self.body_runtime = body_runtime
        self.interval_s = max(0.2, float(interval_s))
        self.recenter_after_misses = max(1, int(recenter_after_misses))
        self.source = source
        self.engagement_reader = engagement_reader
        self.sleeping_interval_s = max(self.interval_s, float(sleeping_interval_s))
        self.session_id = "tracking-session"
        self.actor_id = "vision-runtime"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._tracking_paused = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="eibrain-visual-tracking", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                should_track = self._should_track()
                if should_track:
                    self._tracking_paused = False
                    self._track_once()
                else:
                    self._pause_tracking_once(reason="engagement_inactive")
            except Exception as exc:  # pragma: no cover - hardware boundary
                record = getattr(self.body_runtime, "record_runtime_event", None)
                if callable(record):
                    record(
                        kind="visual_tracking_error",
                        source="eye.tracking",
                        status="error",
                        details={"error": str(exc)},
                    )
            elapsed = time.monotonic() - started
            interval_s = self.interval_s if should_track else self.sleeping_interval_s
            self._stop.wait(max(0.0, interval_s - elapsed))

    def _should_track(self) -> bool:
        if self.engagement_reader is None:
            return True
        return self.engagement_reader.should_run_vision()

    def _track_once(self) -> None:
        track = self.body_runtime.track_visual_target_once
        kwargs = {"session_id": self.session_id, "actor_id": self.actor_id}
        signature = inspect.signature(track)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        if accepts_kwargs or "recenter_after_misses" in signature.parameters:
            kwargs["recenter_after_misses"] = self.recenter_after_misses
        if accepts_kwargs or "source" in signature.parameters:
            kwargs["source"] = self.source
        track(**kwargs)

    def _pause_tracking_once(self, *, reason: str) -> None:
        if self._tracking_paused:
            return
        pause = getattr(self.body_runtime, "pause_visual_tracking", None)
        if callable(pause):
            pause(reason=reason)
        self._tracking_paused = True

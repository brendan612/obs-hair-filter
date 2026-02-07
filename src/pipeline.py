import cv2
import numpy as np
import time
import threading
from typing import Callable, Optional

from src.capture import ThreadedCamera
from src.segmenter import HairSegmenter
from src.color_transform import ColorTransformer
from src.virtual_camera import VirtualCameraOutput
from src.config import AppConfig


class AsyncSegmenter:
    """
    Runs hair segmentation in a dedicated thread.
    The main pipeline loop grabs the latest mask without waiting,
    so segmentation and color transform run in parallel.
    """

    def __init__(self, segmenter: HairSegmenter, scale: float):
        self._segmenter = segmenter
        self._scale = scale
        self._lock = threading.Lock()
        self._latest_mask: Optional[np.ndarray] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_event = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray):
        """Submit a new frame for segmentation (non-blocking)."""
        with self._lock:
            self._latest_frame = frame
        self._new_frame_event.set()

    def get_mask(self) -> Optional[np.ndarray]:
        """Get the most recent segmentation mask (non-blocking)."""
        with self._lock:
            return self._latest_mask

    def _loop(self):
        while self._running:
            # Wait for a new frame (with timeout so we can check _running)
            self._new_frame_event.wait(timeout=0.1)
            self._new_frame_event.clear()

            with self._lock:
                frame = self._latest_frame
            if frame is None:
                continue

            h, w = frame.shape[:2]
            scale = self._scale

            # Downscale
            if scale < 1.0:
                proc_w, proc_h = int(w * scale), int(h * scale)
                frame_small = cv2.resize(
                    frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                frame_small = frame

            # Run segmentation
            mask_small = self._segmenter.get_hair_mask(frame_small)

            # Upscale mask
            if scale < 1.0:
                mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                mask = mask_small

            with self._lock:
                self._latest_mask = mask

    def stop(self):
        self._running = False
        self._new_frame_event.set()  # Unblock the wait
        if self._thread:
            self._thread.join(timeout=2.0)


class Pipeline:
    """
    Orchestrates the full processing pipeline:
    Camera -> Async Segmentation -> Color Transform -> Virtual Camera

    Threading model:
    - Camera I/O: background thread (ThreadedCamera)
    - Segmentation: background thread (AsyncSegmenter) — runs in parallel
    - Main loop: color transform + output (this thread)
    - The main loop uses the latest available mask, so it never waits
      for segmentation. This means color transform and segmentation
      overlap, and FPS is limited by the slower of the two, not their sum.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._running = False

        self._camera: Optional[ThreadedCamera] = None
        self._segmenter: Optional[HairSegmenter] = None
        self._async_seg: Optional[AsyncSegmenter] = None
        self._transformer: Optional[ColorTransformer] = None
        self._vcam: Optional[VirtualCameraOutput] = None

        # GUI preview callback: receives BGR frame
        self.on_frame: Optional[Callable[[np.ndarray], None]] = None

        # FPS tracking
        self._fps_counter = 0
        self._fps_time = time.time()
        self.current_fps = 0.0

        # Per-stage timing (rolling averages in ms)
        self._timing = {"color": 0.0, "total": 0.0}

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Initialize all components and prepare for processing."""
        cfg = self.config

        # Camera capture thread
        self._camera = ThreadedCamera(cfg.camera_index, cfg.width, cfg.height)
        self._camera.start()

        actual_w, actual_h = self._camera.resolution
        print(f"Camera actual resolution: {actual_w}x{actual_h}")

        # Hair segmenter + async wrapper
        self._segmenter = HairSegmenter(
            model_path=cfg.model_path,
            confidence_threshold=cfg.mask_threshold,
            blur_kernel=cfg.mask_blur_kernel,
            temporal_alpha=cfg.temporal_alpha,
        )
        self._async_seg = AsyncSegmenter(self._segmenter, cfg.processing_scale)
        self._async_seg.start()

        # Color transformer
        self._transformer = ColorTransformer()
        self._transformer.set_target_color_lab(*cfg.target_color_lab)
        self._transformer.set_intensity(cfg.color_intensity)

        # Virtual camera output
        self._vcam = VirtualCameraOutput(
            cfg.width, cfg.height, cfg.fps, cfg.vcam_backend
        )
        device = self._vcam.start()
        print(f"Virtual camera started: {device}")
        print(f"Resolution: {cfg.width}x{cfg.height} @ {cfg.fps}fps")
        print(f"Processing scale: {cfg.processing_scale}")

        self._running = True

    def run_loop(self):
        """
        Main processing loop. Call from a background thread.
        Segmentation runs async — we always use the latest available mask.
        """
        while self._running:
            t_start = time.perf_counter()

            frame = self._camera.read()
            if frame is None:
                time.sleep(0.001)
                continue

            # Submit frame for async segmentation (non-blocking)
            self._async_seg.submit_frame(frame)

            # Get latest mask (may be from a previous frame — that's OK)
            mask = self._async_seg.get_mask()

            # Color transformation
            t2 = time.perf_counter()
            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                result = self._transformer.apply(frame, mask)
            else:
                # No mask yet (first few frames) — pass through unmodified
                result = frame
            t3 = time.perf_counter()

            # Output to virtual camera
            self._vcam.send_frame(result)

            # GUI preview callback
            if self.on_frame:
                self.on_frame(result)

            t_end = time.perf_counter()

            # Rolling average timing (ms)
            alpha = 0.1
            self._timing["color"] = (1 - alpha) * self._timing["color"] + alpha * (t3 - t2) * 1000
            self._timing["total"] = (1 - alpha) * self._timing["total"] + alpha * (t_end - t_start) * 1000

            # FPS tracking
            self._fps_counter += 1
            now = time.time()
            elapsed = now - self._fps_time
            if elapsed >= 1.0:
                self.current_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_time = now
                print(
                    f"FPS: {self.current_fps:.1f} | "
                    f"Color: {self._timing['color']:.1f}ms | "
                    f"Total: {self._timing['total']:.1f}ms"
                )

            # No sleep — run as fast as possible for minimum latency.
            # The virtual camera handles frame timing internally.

    def update_color(self, l: int, a: int, b: int, intensity: Optional[float] = None):
        """Live update of target color. Thread-safe (called from GUI thread)."""
        if self._transformer:
            self._transformer.set_target_color_lab(l, a, b)
            if intensity is not None:
                self._transformer.set_intensity(intensity)

    def update_color_rgb(self, r: int, g: int, b: int, intensity: Optional[float] = None):
        """Live update of target color from RGB. Thread-safe."""
        if self._transformer:
            self._transformer.set_target_color_rgb(r, g, b)
            if intensity is not None:
                self._transformer.set_intensity(intensity)

    def update_segmenter(
        self,
        temporal_alpha: Optional[float] = None,
        blur_kernel: Optional[int] = None,
    ):
        """Live update of segmenter parameters. Thread-safe."""
        if self._segmenter:
            if temporal_alpha is not None:
                self._segmenter.temporal_alpha = temporal_alpha
            if blur_kernel is not None:
                self._segmenter.blur_kernel = blur_kernel

    def stop(self):
        """Stop all components and clean up."""
        self._running = False
        if self._async_seg:
            self._async_seg.stop()
            self._async_seg = None
        if self._camera:
            self._camera.stop()
            self._camera = None
        if self._segmenter:
            self._segmenter.close()
            self._segmenter = None
        if self._vcam:
            self._vcam.stop()
            self._vcam = None

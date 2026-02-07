import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from typing import Optional
import cv2


class VirtualCameraOutput:
    """
    Wraps pyvirtualcam to send processed frames to a virtual camera
    that OBS can capture as a video source.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        backend: str = "obs",
    ):
        self._width = width
        self._height = height
        self._fps = fps
        self._backend = backend
        self._cam: Optional[pyvirtualcam.Camera] = None

    @property
    def device(self) -> Optional[str]:
        return self._cam.device if self._cam else None

    def start(self) -> str:
        """Start virtual camera. Returns the device name."""
        self._cam = pyvirtualcam.Camera(
            width=self._width,
            height=self._height,
            fps=self._fps,
            fmt=PixelFormat.BGR,
            backend=self._backend,
        )
        return self._cam.device

    def send_frame(self, frame_bgr: np.ndarray):
        """Send a BGR frame to the virtual camera."""
        if self._cam is None:
            return
        h, w = frame_bgr.shape[:2]
        if w != self._width or h != self._height:
            frame_bgr = cv2.resize(frame_bgr, (self._width, self._height))
        self._cam.send(frame_bgr)

    def sleep_until_next(self):
        """Adaptive sleep to maintain target FPS."""
        if self._cam:
            self._cam.sleep_until_next_frame()

    def stop(self):
        if self._cam:
            self._cam.close()
            self._cam = None

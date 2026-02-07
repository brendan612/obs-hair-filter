import cv2
import threading
import time
import numpy as np
from typing import Optional, Tuple


class ThreadedCamera:
    """
    Captures frames from webcam in a background thread.
    Uses grab/retrieve pattern to drain the driver buffer and
    always provide the freshest possible frame.
    """

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720):
        self._cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Try to set higher FPS from the camera itself
        self._cap.set(cv2.CAP_PROP_FPS, 60)

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def resolution(self) -> Tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def start(self) -> "ThreadedCamera":
        if self._running:
            return self
        if not self._cap.isOpened():
            raise RuntimeError(
                "Cannot open camera. Check that your webcam is connected "
                "and not in use by another application."
            )
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self):
        """
        Continuously grab frames as fast as possible.
        grab() is much faster than read() — it just advances the buffer
        without decoding. We only decode (retrieve) the latest one.
        This drains any stale frames the driver has buffered.
        """
        while self._running:
            # Grab (fast, no decode) — drains the driver buffer
            if not self._cap.grab():
                time.sleep(0.001)
                continue

            # Retrieve (decode) only the most recent grabbed frame
            ret, frame = self._cap.retrieve()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        """Returns the most recent frame (BGR), or None if no frame yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap.isOpened():
            self._cap.release()

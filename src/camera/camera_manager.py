"""
src/camera/camera_manager.py
=============================
Encapsulates all OpenCV camera concerns.
Returns BGR frames and handles open/close lifecycle.
"""

import cv2
from typing import Optional, Tuple
import numpy as np

from src.core.config import CONFIG


class CameraError(RuntimeError):
    pass


class CameraManager:
    """Thread-safe (single-thread) camera wrapper."""

    def __init__(self):
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        cfg = CONFIG.camera
        self._cap = cv2.VideoCapture(cfg.device_index)
        if not self._cap.isOpened():
            raise CameraError(f"Cannot open camera device {cfg.device_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        # Reduce internal buffer to 1 frame → lower latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame. Returns (ok, bgr_frame)."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ok, frame = self._cap.read()
        if ok:
            frame = cv2.flip(frame, 1)   # mirror so gestures feel natural
        return ok, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()

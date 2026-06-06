"""
src/gestures/palm_detector.py
================================
Open-palm detection.

A palm is open when ALL four fingers (index, middle, ring, pinky) have their
tip landmark ABOVE (smaller y value) their corresponding PIP joint.
Thumb is excluded (its geometry is orientation-dependent).

Confidence is computed as the fraction of fingers that satisfy the condition.
At 1.0 → all 4 fingers extended.  Threshold configurable in CONFIG.
"""

from src.core.config import CONFIG
from src.core.app_state import AppState


# MediaPipe landmark indices
_FINGERS = [
    (8,  6),   # index:  tip, pip
    (12, 10),  # middle: tip, pip
    (16, 14),  # ring:   tip, pip
    (20, 18),  # pinky:  tip, pip
]


class PalmDetector:

    def __init__(self):
        self._threshold = CONFIG.palm.confidence_threshold

    def update(self, state: AppState) -> None:
        """
        Updates state.auto_rotate based on open-palm detection.
        Reads state._landmarks (set by HandTracker each frame).
        """
        landmarks = getattr(state, "_landmarks", None)
        if landmarks is None:
            state.auto_rotate = False
            return

        extended = 0
        for tip_idx, pip_idx in _FINGERS:
            tip_y = landmarks[tip_idx].y
            pip_y = landmarks[pip_idx].y
            if tip_y < pip_y:   # tip is higher on screen = finger extended
                extended += 1

        confidence = extended / len(_FINGERS)
        state.auto_rotate = (confidence >= self._threshold)

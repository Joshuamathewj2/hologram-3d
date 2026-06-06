"""
src/gestures/pinch_detector.py
================================
Pinch detection with hysteresis to prevent rapid on/off toggling.

Uses filtered pinch_dist from HandData (already smoothed by EMA filter).

Hysteresis:
  active_threshold  < release_threshold
  → pinch activates at 0.06, releases at 0.10
  This prevents flickering when hand hovers near the boundary.

Side effect: maps pinch distance → zoom value (linear interpolation).
"""

import math
from src.core.config import CONFIG
from src.core.app_state import AppState, GestureState


class PinchDetector:

    def __init__(self):
        cfg = CONFIG.pinch
        self._active_thresh  = cfg.active_threshold
        self._release_thresh = cfg.release_threshold
        self._zoom_min  = cfg.zoom_min
        self._zoom_max  = cfg.zoom_max
        self._pd_min    = cfg.pinch_dist_min
        self._pd_max    = cfg.pinch_dist_max

    def update(self, state: AppState) -> None:
        """
        Reads state.hand.pinch_dist, updates state.pinch_active + state.zoom.
        Gesture state transitions are handled by the state machine in main, but
        this detector exposes pinch_active as a flag the machine reads.
        """
        dist = state.hand.pinch_dist

        # Hysteretic toggle
        if not state.pinch_active:
            if dist < self._active_thresh:
                state.pinch_active = True
        else:
            if dist > self._release_thresh:
                state.pinch_active = False

        # Always update zoom based on pinch distance (continuous mapping)
        # Clamp into range first
        d = max(self._pd_min, min(self._pd_max, dist))
        t = (d - self._pd_min) / (self._pd_max - self._pd_min)
        # t=0 → close pinch → zoom_max (-3, near camera)
        # t=1 → open pinch  → zoom_min (-12, far camera)
        state.zoom = self._zoom_max + t * (self._zoom_min - self._zoom_max)

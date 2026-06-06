"""
tests/test_pinch_detector.py
==============================
Unit tests for PinchDetector (hysteresis + zoom mapping).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.core.app_state import AppState, HandVisibility, GestureState
from src.gestures.pinch_detector import PinchDetector
from src.core.config import CONFIG


def _state(pinch_dist: float) -> AppState:
    s = AppState()
    s.hand_visibility = HandVisibility.VISIBLE
    s.hand.pinch_dist = pinch_dist
    return s


class TestPinchActivation:

    def test_activates_below_threshold(self):
        det = PinchDetector()
        s   = _state(0.04)   # below active_threshold (0.06)
        det.update(s)
        assert s.pinch_active is True

    def test_does_not_activate_above_threshold(self):
        det = PinchDetector()
        s   = _state(0.12)   # above active_threshold
        det.update(s)
        assert s.pinch_active is False

    def test_hysteresis_stays_active_until_release(self):
        """Pinch activated at 0.04 should remain active at 0.08 (< release 0.10)."""
        det = PinchDetector()
        s   = _state(0.04)
        det.update(s)
        assert s.pinch_active is True

        s.hand.pinch_dist = 0.08   # below release threshold
        det.update(s)
        assert s.pinch_active is True   # still active

    def test_hysteresis_releases_above_release_threshold(self):
        """Pinch should release when distance exceeds release_threshold (0.10)."""
        det = PinchDetector()
        s   = _state(0.04)
        det.update(s)
        assert s.pinch_active is True

        s.hand.pinch_dist = 0.12   # above release threshold
        det.update(s)
        assert s.pinch_active is False


class TestZoomMapping:

    def test_close_pinch_zooms_in(self):
        """Small dist → zoom_max (closer to camera)."""
        cfg = CONFIG.pinch
        det = PinchDetector()
        s   = _state(cfg.pinch_dist_min)
        det.update(s)
        assert abs(s.zoom - cfg.zoom_max) < 0.1

    def test_open_pinch_zooms_out(self):
        """Large dist → zoom_min (far from camera)."""
        cfg = CONFIG.pinch
        det = PinchDetector()
        s   = _state(cfg.pinch_dist_max)
        det.update(s)
        assert abs(s.zoom - cfg.zoom_min) < 0.1

    def test_zoom_monotone(self):
        """Increasing distance should monotonically decrease zoom (move camera back)."""
        det = PinchDetector()
        dists = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]
        zooms = []
        for d in dists:
            s = _state(d)
            det.update(s)
            zooms.append(s.zoom)
        # Each zoom should be less than or equal to previous (farther = more negative)
        for i in range(1, len(zooms)):
            assert zooms[i] <= zooms[i - 1] + 0.001  # allow small float noise

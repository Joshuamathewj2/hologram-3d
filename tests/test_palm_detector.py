"""
tests/test_palm_detector.py
=============================
Unit tests for PalmDetector.
Uses mock landmark objects to avoid needing MediaPipe installed.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.core.app_state import AppState, HandVisibility
from src.gestures.palm_detector import PalmDetector


# ---------------------------------------------------------------------------
# Minimal landmark stub
# ---------------------------------------------------------------------------

class _LM:
    def __init__(self, y: float):
        self.y = y


def _make_landmarks(
    idx_tip_y=0.3, idx_pip_y=0.5,
    mid_tip_y=0.3, mid_pip_y=0.5,
    rng_tip_y=0.3, rng_pip_y=0.5,
    pky_tip_y=0.3, pky_pip_y=0.5,
):
    """Build a 21-element fake landmark list. Only indices 8,6,12,10,16,14,20,18 matter."""
    lm = [_LM(0.5)] * 21
    lm[8]  = _LM(idx_tip_y)
    lm[6]  = _LM(idx_pip_y)
    lm[12] = _LM(mid_tip_y)
    lm[10] = _LM(mid_pip_y)
    lm[16] = _LM(rng_tip_y)
    lm[14] = _LM(rng_pip_y)
    lm[20] = _LM(pky_tip_y)
    lm[18] = _LM(pky_pip_y)
    return lm


def _state_with_landmarks(landmarks) -> AppState:
    s = AppState()
    s.hand_visibility = HandVisibility.VISIBLE
    s._landmarks = landmarks
    return s


class TestOpenPalm:

    def test_all_extended_triggers_auto_rotate(self):
        """All 4 fingers extended → auto_rotate = True."""
        lm = _make_landmarks(
            idx_tip_y=0.2, idx_pip_y=0.5,   # tip above pip
            mid_tip_y=0.2, mid_pip_y=0.5,
            rng_tip_y=0.2, rng_pip_y=0.5,
            pky_tip_y=0.2, pky_pip_y=0.5,
        )
        det = PalmDetector()
        s   = _state_with_landmarks(lm)
        det.update(s)
        assert s.auto_rotate is True

    def test_all_curled_no_auto_rotate(self):
        """All fingers curled → auto_rotate = False."""
        lm = _make_landmarks(
            idx_tip_y=0.8, idx_pip_y=0.5,
            mid_tip_y=0.8, mid_pip_y=0.5,
            rng_tip_y=0.8, rng_pip_y=0.5,
            pky_tip_y=0.8, pky_pip_y=0.5,
        )
        det = PalmDetector()
        s   = _state_with_landmarks(lm)
        det.update(s)
        assert s.auto_rotate is False

    def test_three_fingers_below_threshold(self):
        """3/4 = 75% < 85% threshold → should NOT trigger."""
        lm = _make_landmarks(
            idx_tip_y=0.2, idx_pip_y=0.5,
            mid_tip_y=0.2, mid_pip_y=0.5,
            rng_tip_y=0.2, rng_pip_y=0.5,
            pky_tip_y=0.8, pky_pip_y=0.5,   # pinky curled
        )
        det = PalmDetector()
        s   = _state_with_landmarks(lm)
        det.update(s)
        assert s.auto_rotate is False

    def test_no_landmarks_no_auto_rotate(self):
        """Missing _landmarks attribute → graceful False."""
        det = PalmDetector()
        s   = AppState()
        det.update(s)
        assert s.auto_rotate is False

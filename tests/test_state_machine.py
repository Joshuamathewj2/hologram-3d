"""
tests/test_state_machine.py
=============================
Unit tests for GestureStateMachine and related state transitions.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.core.app_state import AppState, GestureState, HandVisibility
from src.gestures.state_machine import GestureStateMachine


def _state(
    visible: bool = True,
    pinch: bool = False,
    palm: bool = False,
    swipe_vel: float = 0.0,
) -> AppState:
    s = AppState()
    s.hand_visibility = HandVisibility.VISIBLE if visible else HandVisibility.LOST
    s.pinch_active    = pinch
    s.auto_rotate     = palm
    s.swipe_velocity  = swipe_vel
    return s


class TestHandNotVisible:

    def test_idle_when_hand_lost(self):
        sm = GestureStateMachine()
        s  = _state(visible=False)
        sm.update(s)
        assert s.gesture_state == GestureState.IDLE

    def test_auto_rotate_off_when_hand_lost(self):
        sm = GestureStateMachine()
        s  = _state(visible=False, palm=True)
        sm.update(s)
        assert s.auto_rotate is False


class TestNormalTracking:

    def test_tracking_when_hand_visible_no_gesture(self):
        sm = GestureStateMachine()
        s  = _state()
        sm.update(s)
        assert s.gesture_state == GestureState.TRACKING


class TestPinchPriority:

    def test_pinching_state(self):
        sm = GestureStateMachine()
        s  = _state(pinch=True)
        sm.update(s)
        assert s.gesture_state == GestureState.PINCHING

    def test_pinch_blocks_palm(self):
        """Pinch has higher priority than palm → should be PINCHING, not OPEN_PALM."""
        sm = GestureStateMachine()
        s  = _state(pinch=True, palm=True)
        sm.update(s)
        assert s.gesture_state == GestureState.PINCHING
        assert s.auto_rotate is False


class TestSwipePriority:

    def test_swiping_state_highest_priority(self):
        """Swipe beats everything."""
        sm = GestureStateMachine()
        s  = _state(pinch=True, palm=True, swipe_vel=0.5)
        sm.update(s)
        assert s.gesture_state == GestureState.SWIPING

    def test_swipe_velocity_zeroed_after(self):
        """State machine clears swipe_velocity to prevent perpetual SWIPING."""
        sm = GestureStateMachine()
        s  = _state(swipe_vel=0.7)
        sm.update(s)
        assert s.swipe_velocity == 0.0


class TestOpenPalmState:

    def test_open_palm_state(self):
        sm = GestureStateMachine()
        s  = _state(palm=True)
        sm.update(s)
        assert s.gesture_state == GestureState.OPEN_PALM

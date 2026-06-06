"""
src/gestures/state_machine.py
================================
Gesture interaction state machine.

States
------
IDLE       → no hand visible
TRACKING   → hand visible, no active gesture
PINCHING   → pinch distance below threshold (exclusive from SWIPING)
SWIPING    → swipe motion detected (exclusive from PINCHING)
OPEN_PALM  → all fingers extended (exclusive from PINCHING/SWIPING)

Priorities (highest wins when overlapping):
  SWIPING > PINCHING > OPEN_PALM > TRACKING > IDLE

This prevents:
- Simultaneous pinch + swipe (physically impossible but noisy detection can
  produce both signals at once)
- False palm during pinch (palm requires open fingers — but just in case)
- Swipe while pinching (intentional: don't switch objects while user is zooming)

Transitions
-----------
IDLE
  hand appears → TRACKING

TRACKING
  swipe detected → SWIPING (temporarily; returns to TRACKING after)
  pinch active   → PINCHING
  palm active    → OPEN_PALM

PINCHING
  pinch released → TRACKING
  swipe detected → blocked (swipe ignored while pinching)

OPEN_PALM
  palm gone      → TRACKING
  pinch active   → PINCHING (higher priority)

SWIPING
  after swipe accepted → TRACKING immediately (swipe is instantaneous)
"""

import time
from src.core.app_state import AppState, GestureState, HandVisibility
from src.core.config import CONFIG


class GestureStateMachine:

    def update(self, state: AppState) -> None:
        """
        Evaluate current sensor signals and advance state machine.
        Must be called AFTER all detectors have updated their flags on state.
        """
        vis = state.hand_visibility

        # ── Hand not visible ────────────────────────────────────────────────
        if vis != HandVisibility.VISIBLE:
            state.gesture_state = GestureState.IDLE
            state.auto_rotate = False
            return

        # ── Determine dominant intent ────────────────────────────────────────
        # Swipe is highest priority — it's transient and already accepted
        # (swipe_velocity > 0 for one frame only after acceptance)
        if state.swipe_velocity > 0.0:
            state.gesture_state = GestureState.SWIPING
            # Swiping is instantaneous; next frame will re-evaluate
            state.swipe_velocity = 0.0
            return

        # Pinch blocks OPEN_PALM — can't pinch and have open palm simultaneously
        if state.pinch_active:
            state.gesture_state = GestureState.PINCHING
            state.auto_rotate = False  # palm auto-rotate off during pinch
            return

        # Open palm
        if state.auto_rotate:
            state.gesture_state = GestureState.OPEN_PALM
            return

        # Default: hand visible but no active gesture
        state.gesture_state = GestureState.TRACKING

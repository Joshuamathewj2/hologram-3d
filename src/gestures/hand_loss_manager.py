"""
src/gestures/hand_loss_manager.py
==================================
Handles the lifecycle of hand loss events.

Behaviour timeline after hand disappears:
  0  – grace_period (1 s): grace — state frozen (hand may reappear)
  1  – recovery_start (1 s) after grace: recovery — slow lerp toward neutral
  3+ s: full state reset

This prevents the hologram from freezing permanently when the hand leaves
the camera view.
"""

import time
from src.core.config import CONFIG
from src.core.app_state import AppState, HandVisibility, GestureState


class HandLossManager:

    def __init__(self):
        cfg = CONFIG.hand_loss
        self._grace     = cfg.grace_period
        self._rec_start = cfg.recovery_start
        self._rec_dur   = cfg.recovery_duration
        self._reset_at  = cfg.reset_after

    def update(self, state: AppState, dt: float) -> None:
        """
        Call every frame.  Manages state.hand_visibility transitions
        and neutral-pose recovery when hand is absent.
        """
        if state.hand_visibility == HandVisibility.VISIBLE:
            return   # nothing to do

        if state.hand_lost_time is None:
            return   # hand was never seen

        elapsed = time.time() - state.hand_lost_time

        # ── Full reset ───────────────────────────────────────────────────────
        if elapsed >= self._reset_at:
            state.hand_visibility  = HandVisibility.LOST
            state.gesture_state    = GestureState.IDLE
            state.auto_rotate      = False
            state.pinch_active     = False
            state.swipe_velocity   = 0.0
            state.rot_x            = state.rot_x_neutral
            state.rot_y            = state.rot_y_neutral
            return

        # ── Recovery lerp ────────────────────────────────────────────────────
        if elapsed >= self._grace + self._rec_start:
            state.hand_visibility = HandVisibility.LOST
            # Lerp fraction this frame
            frac_elapsed = elapsed - (self._grace + self._rec_start)
            t = min(1.0, frac_elapsed / self._rec_dur)
            state.rot_x = _lerp(state.rot_x, state.rot_x_neutral, t * dt * 3.0)
            state.rot_y = _lerp(state.rot_y, state.rot_y_neutral, t * dt * 3.0)
            return

        # ── Grace period ─────────────────────────────────────────────────────
        state.hand_visibility = HandVisibility.GRACE
        # State frozen — nothing changes


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

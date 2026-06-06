"""
src/gestures/swipe_detector.py
================================
Genuine swipe detection using temporal position history.

HOW IT WORKS
------------
Previous (fake) system:
    move = index_tip.x - wrist.x
    if move > 0.20: swipe right

This measured a STATIC POSE (index tip to the right of wrist) — it was not
motion at all.  It triggered on *any* right-pointing hand pose, continuously.

Real swipe detection:
1. Maintain a rolling deque of (timestamp, x_position) samples.
2. On each frame, look for a completed gesture window:
   - displacement: |x_end - x_start| > threshold
   - velocity:     displacement / duration  > threshold
   - duration:     within [min, max] seconds
   - cooldown:     not too soon after the last accepted swipe
3. Only clear the deque when a swipe is accepted, so partial strokes
   that fail the test keep accumulating until they either succeed or
   age out naturally via `maxlen`.

NOTE ON COOLDOWN
----------------
Cooldown is compared against the timestamp of the *latest sample* (which
equals time.time() at the moment update() was called).  This means tests
that monkeypatch time.time() get a consistent, controllable clock throughout
the detector — no wall-clock leakage.

PUBLIC API
----------
update(state) -> None
    Call every frame when hand is VISIBLE.
    Mutates state.obj_index on completion.
    Sets state.swipe_velocity for the debug HUD.
"""

import time
from collections import deque
from typing import Deque, List, Optional, Tuple

from src.core.config import CONFIG
from src.core.app_state import AppState


# ---------------------------------------------------------------------------
# Internal sample record
# ---------------------------------------------------------------------------

class _Sample:
    __slots__ = ("t", "x")
    def __init__(self, t: float, x: float):
        self.t = t
        self.x = x


# ---------------------------------------------------------------------------
# SwipeDetector
# ---------------------------------------------------------------------------

class SwipeDetector:
    """
    Stateful swipe recogniser.  One instance per session.
    Must be reset() when the hand is lost.
    """

    def __init__(self):
        cfg = CONFIG.swipe
        self._cfg = cfg
        self._history: Deque[_Sample] = deque(maxlen=cfg.history_len)
        # Initialise far in the past so first swipe is never blocked by cooldown
        self._last_accepted: float = -10_000.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call when hand disappears so stale history doesn't pollute next session."""
        self._history.clear()

    def update(self, state: AppState) -> None:
        """
        Ingest current index-tip x position and attempt to recognise a swipe.
        Mutates state in place.
        """
        now = time.time()   # tests may monkeypatch this
        x   = state.hand.index_x

        self._history.append(_Sample(now, x))

        # Need at least 3 samples to compute anything meaningful
        if len(self._history) < 3:
            return

        # Cooldown guard — use the timestamp from the latest *sample* so
        # that a monkeypatched time.time() provides a consistent clock here.
        latest_t = self._history[-1].t
        if latest_t - self._last_accepted < self._cfg.cooldown:
            return

        result = self._analyse()
        if result is None:
            state.swipe_velocity = 0.0
            return

        direction, velocity, accept_t = result
        state.swipe_velocity = velocity
        self._last_accepted = accept_t
        self._history.clear()

        if direction == "RIGHT":
            state.next_object()
            state.mark_swipe("RIGHT")
        else:
            state.prev_object()
            state.mark_swipe("LEFT")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _analyse(self) -> Optional[Tuple[str, float, float]]:
        """
        Scan history for the *best* qualifying swipe window.

        Returns (direction, velocity, accept_timestamp) or None.
        """
        cfg     = self._cfg
        samples = list(self._history)

        for i in range(len(samples) - 2):
            s = samples[i]
            for j in range(i + 2, len(samples)):
                e = samples[j]

                duration = e.t - s.t
                if duration < cfg.min_duration:
                    continue
                if duration > cfg.max_duration:
                    break   # this start is too old; try newer start

                displacement = e.x - s.x
                abs_disp = abs(displacement)

                if abs_disp < cfg.displacement_threshold:
                    continue

                velocity = abs_disp / duration
                if velocity < cfg.velocity_threshold:
                    continue

                segment = samples[i : j + 1]
                if not self._is_monotone(segment, displacement):
                    continue

                direction = "RIGHT" if displacement > 0 else "LEFT"
                return direction, velocity, e.t

        return None

    @staticmethod
    def _is_monotone(samples: List[_Sample], net_displacement: float) -> bool:
        """
        Verify the gesture contains no intermediate reversal larger than 20%
        of the total peak-to-peak range.

        For a RIGHT swipe:
            - The path moves generally left→right.
            - Reversal = any point that goes *below* the starting x.
        For a LEFT swipe:
            - Reversal = any point that goes *above* the starting x.
        """
        if len(samples) < 2:
            return False

        xs          = [s.x for s in samples]
        x_start     = xs[0]
        x_min       = min(xs)
        x_max       = max(xs)
        total_range = x_max - x_min
        if total_range == 0:
            return False

        if net_displacement > 0:
            # Right swipe — penalise any reversal below the start
            max_reversal = max(0.0, *(x_start - x for x in xs))
        else:
            # Left swipe — penalise any excursion above the start
            max_reversal = max(0.0, *(x - x_start for x in xs))

        return (max_reversal / total_range) < 0.20

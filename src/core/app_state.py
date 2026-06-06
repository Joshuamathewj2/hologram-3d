"""
src/core/app_state.py
=====================
Single source of truth for all mutable application state.
No global variables anywhere else — everything flows through AppState.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GestureState(Enum):
    IDLE       = auto()   # no hand / no gesture
    TRACKING   = auto()   # hand visible, no active gesture
    PINCHING   = auto()   # pinch active
    SWIPING    = auto()   # swipe in progress
    OPEN_PALM  = auto()   # open palm / auto-rotate active


class HandVisibility(Enum):
    VISIBLE = auto()
    GRACE   = auto()   # recently lost (0–1 s)
    LOST    = auto()   # definitely absent (>1 s), recovering


# ---------------------------------------------------------------------------
# Tracked-hand data (filled each frame)
# ---------------------------------------------------------------------------

@dataclass
class HandData:
    """Filtered landmark positions for the currently tracked hand."""
    index_x: float = 0.5
    index_y: float = 0.5
    thumb_x: float = 0.5
    thumb_y: float = 0.5
    wrist_x: float = 0.5
    wrist_y: float = 0.5

    # Raw (unfiltered) equivalents, kept for debug overlay
    raw_index_x: float = 0.5
    raw_index_y: float = 0.5
    raw_thumb_x: float = 0.5
    raw_thumb_y: float = 0.5

    # Pinch distance (filtered)
    pinch_dist: float = 0.15

    # Detection confidence reported by MediaPipe
    hand_confidence: float = 0.0
    tracking_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Main application state
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    # ── Gesture machine ─────────────────────────────────────────────────────
    gesture_state: GestureState = GestureState.IDLE
    hand_visibility: HandVisibility = HandVisibility.LOST

    # ── Object selection ────────────────────────────────────────────────────
    obj_index: int = 0
    obj_count: int = 8

    # ── Rotation (degrees) ──────────────────────────────────────────────────
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_x_neutral: float = 0.0    # target during recovery
    rot_y_neutral: float = 0.0

    # ── Zoom ────────────────────────────────────────────────────────────────
    zoom: float = -8.0

    # ── Auto-rotate ─────────────────────────────────────────────────────────
    auto_rotate: bool = False
    auto_rotate_speed: float = 45.0   # deg/s (frame-rate independent)

    # ── Timing ──────────────────────────────────────────────────────────────
    last_gesture_time: float = field(default_factory=time.time)
    hand_lost_time: Optional[float] = None   # timestamp when hand vanished
    last_frame_time: float = field(default_factory=time.time)

    # ── Hand data ───────────────────────────────────────────────────────────
    hand: HandData = field(default_factory=HandData)

    # ── Debug ───────────────────────────────────────────────────────────────
    debug_overlay: bool = False
    fps: float = 0.0
    swipe_velocity: float = 0.0
    frame_count: int = 0

    # ── Swipe feedback ──────────────────────────────────────────────────────
    last_swipe_direction: str = ""    # "LEFT" | "RIGHT" | ""
    swipe_flash_until: float = 0.0    # time until swipe label disappears

    # ── Pinch feedback ──────────────────────────────────────────────────────
    pinch_active: bool = False

    def delta_time(self) -> float:
        """Return elapsed seconds since last frame; update timer."""
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        # Safety clamp: never let dt exceed 100 ms (avoids spiral after pause)
        return min(dt, 0.1)

    def next_object(self):
        self.obj_index = (self.obj_index + 1) % self.obj_count

    def prev_object(self):
        self.obj_index = (self.obj_index - 1) % self.obj_count

    def mark_swipe(self, direction: str):
        self.last_swipe_direction = direction
        self.swipe_flash_until = time.time() + 1.5

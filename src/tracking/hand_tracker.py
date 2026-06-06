"""
src/tracking/hand_tracker.py
==============================
MediaPipe hand detection + Exponential Moving Average (EMA) landmark filter.

WHY EMA?
--------
Raw MediaPipe coordinates oscillate ±0.01–0.03 normalised units per frame
(visible as jitter / shaking). EMA applies per-landmark low-pass smoothing:

    filtered = alpha * raw + (1 - alpha) * filtered_prev

alpha = 0.25  →  strong smoothing, ~4-frame lag at 30 fps (imperceptible)
alpha = 1.00  →  no smoothing (raw pass-through)

The filtered coordinates are stored in HandData. The raw coordinates are
kept alongside for the debug overlay comparison.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.core.config import CONFIG
from src.core.app_state import AppState, HandData, HandVisibility, GestureState


class HandTracker:
    """
    Wraps MediaPipe Hands.
    Call process_frame(rgb_frame, state) every frame.
    Mutates state.hand and state.hand_visibility in place.
    """

    # Landmark indices used across the system
    IDX_WRIST      = 0
    IDX_THUMB_TIP  = 4
    IDX_INDEX_TIP  = 8
    IDX_INDEX_PIP  = 6
    IDX_MIDDLE_TIP = 12
    IDX_MIDDLE_PIP = 10
    IDX_RING_TIP   = 16
    IDX_RING_PIP   = 14
    IDX_PINKY_TIP  = 20
    IDX_PINKY_PIP  = 18

    def __init__(self):
        mp_hands = mp.solutions.hands
        cfg = CONFIG.tracking
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._draw = mp.solutions.drawing_utils
        self._mp_hands = mp_hands

        alpha = cfg.ema_alpha
        self._alpha = alpha
        self._one_minus_alpha = 1.0 - alpha

        # EMA state (initialised None → first-frame seed)
        self._f_ix: Optional[float] = None
        self._f_iy: Optional[float] = None
        self._f_tx: Optional[float] = None
        self._f_ty: Optional[float] = None
        self._f_wx: Optional[float] = None
        self._f_wy: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        state: AppState,
    ) -> np.ndarray:
        """
        Run MediaPipe on bgr_frame, mutate state.hand + state.hand_visibility.
        Returns the annotated BGR frame (with skeleton overlay).
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if result.multi_hand_landmarks:
            # Take only the first hand
            lm = result.multi_hand_landmarks[0]

            # Draw skeleton on frame
            self._draw.draw_landmarks(
                bgr_frame, lm, self._mp_hands.HAND_CONNECTIONS
            )

            # Extract raw values
            r_ix = lm.landmark[self.IDX_INDEX_TIP].x
            r_iy = lm.landmark[self.IDX_INDEX_TIP].y
            r_tx = lm.landmark[self.IDX_THUMB_TIP].x
            r_ty = lm.landmark[self.IDX_THUMB_TIP].y
            r_wx = lm.landmark[self.IDX_WRIST].x
            r_wy = lm.landmark[self.IDX_WRIST].y

            # Seed EMA on first detection after loss
            if self._f_ix is None:
                self._f_ix, self._f_iy = r_ix, r_iy
                self._f_tx, self._f_ty = r_tx, r_ty
                self._f_wx, self._f_wy = r_wx, r_wy

            # Apply EMA filter
            self._f_ix = self._ema(self._f_ix, r_ix)
            self._f_iy = self._ema(self._f_iy, r_iy)
            self._f_tx = self._ema(self._f_tx, r_tx)
            self._f_ty = self._ema(self._f_ty, r_ty)
            self._f_wx = self._ema(self._f_wx, r_wx)
            self._f_wy = self._ema(self._f_wy, r_wy)

            import math
            pinch_dist = math.sqrt(
                (self._f_tx - self._f_ix) ** 2 +
                (self._f_ty - self._f_iy) ** 2
            )

            # Confidence values
            hand_conf = 0.0
            if result.multi_handedness:
                score = result.multi_handedness[0].classification[0].score
                hand_conf = float(score)

            # Populate HandData
            state.hand = HandData(
                index_x=self._f_ix,
                index_y=self._f_iy,
                thumb_x=self._f_tx,
                thumb_y=self._f_ty,
                wrist_x=self._f_wx,
                wrist_y=self._f_wy,
                raw_index_x=r_ix,
                raw_index_y=r_iy,
                raw_thumb_x=r_tx,
                raw_thumb_y=r_ty,
                pinch_dist=pinch_dist,
                hand_confidence=hand_conf,
                tracking_confidence=hand_conf,
            )

            # Store full landmark list for gesture detectors
            state._landmarks = lm.landmark

            state.hand_visibility = HandVisibility.VISIBLE
            state.hand_lost_time = None

        else:
            # Hand not detected — mark as lost, reset EMA seed
            if state.hand_visibility == HandVisibility.VISIBLE:
                import time
                state.hand_lost_time = time.time()
                state.hand_visibility = HandVisibility.GRACE

            self._f_ix = None  # force re-seed on next detection

        return bgr_frame

    def close(self) -> None:
        self._hands.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ema(self, prev: float, raw: float) -> float:
        return self._alpha * raw + self._one_minus_alpha * prev

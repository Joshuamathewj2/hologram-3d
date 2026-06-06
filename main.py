"""
main.py
=======
Entry point for the GestureHologramAI application.

Architecture:
  Camera → HandTracker → Detectors → StateMachine → HandLossManager → Renderer

All state flows through a single AppState object.
No global variables.
Frame-rate independent via delta-time.
"""

import sys
import time

import cv2
import pygame
from pygame.locals import K_F1, KEYDOWN, QUIT

from src.core.app_state import AppState
from src.core.config import CONFIG

from src.camera.camera_manager import CameraManager, CameraError
from src.tracking.hand_tracker import HandTracker

from src.gestures.swipe_detector import SwipeDetector
from src.gestures.pinch_detector import PinchDetector
from src.gestures.palm_detector import PalmDetector
from src.gestures.state_machine import GestureStateMachine
from src.gestures.hand_loss_manager import HandLossManager

from src.renderer.hologram_renderer import HologramRenderer
from src.core.app_state import HandVisibility


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self, window: int = 30):
        self._times = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    pygame.init()

    # Initialise renderer (creates GL context)
    renderer = HologramRenderer()
    renderer.init()

    # Initialise camera
    camera = CameraManager()
    try:
        camera.open()
    except CameraError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        pygame.quit()
        return 1

    # Initialise tracker and detectors
    tracker      = HandTracker()
    swipe_det    = SwipeDetector()
    pinch_det    = PinchDetector()
    palm_det     = PalmDetector()
    state_machine = GestureStateMachine()
    loss_manager  = HandLossManager()

    fps_counter = FPSCounter()

    # Application state
    state = AppState()
    state.last_frame_time = time.time()

    print("[INFO] GestureHologramAI started.  Press Q to quit, F1 for debug overlay.")

    # ── Main loop ────────────────────────────────────────────────────────────
    running = True
    while running:

        # ── Delta time ────────────────────────────────────────────────────
        dt = state.delta_time()
        state.fps = fps_counter.tick()
        state.frame_count += 1

        # ── Pygame events ─────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_F1:
                    state.debug_overlay = not state.debug_overlay

        # ── Camera frame ──────────────────────────────────────────────────
        ok, frame = camera.read()
        if not ok:
            print("[WARN] Failed to read camera frame — skipping.")
            continue

        # ── Hand tracking + EMA filter ────────────────────────────────────
        frame = tracker.process_frame(frame, state)

        # ── Gesture detection (only when hand visible) ────────────────────
        if state.hand_visibility == HandVisibility.VISIBLE:
            pinch_det.update(state)
            palm_det.update(state)
            # Swipe detector must run AFTER pinch so swipe is blocked during pinch
            if not state.pinch_active:
                swipe_det.update(state)
            else:
                swipe_det.reset()  # clear history so pinch→release doesn't misfire

        # ── Hand loss recovery ────────────────────────────────────────────
        loss_manager.update(state, dt)

        # ── Gesture state machine ─────────────────────────────────────────
        state_machine.update(state)

        # ── Render hologram ───────────────────────────────────────────────
        renderer.render(state, dt)

        # ── Camera preview window ─────────────────────────────────────────
        cv2.putText(
            frame, "GestureHologramAI [Q=quit  F1=debug]",
            (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("[INFO] Shutting down…")
    tracker.close()
    camera.release()
    renderer.cleanup()
    cv2.destroyAllWindows()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())

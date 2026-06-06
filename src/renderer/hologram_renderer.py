"""
src/renderer/hologram_renderer.py
==================================
Owns the Pygame/OpenGL window and renders each frame.

Responsibilities:
- Pygame/OpenGL window lifecycle
- Projection + model-view matrix setup
- Frame-rate independent rotation (time-based, not frame-based)
- Delegating object draw to the object catalogue
- Calling the HUD overlay
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from src.core.config import CONFIG
from src.core.app_state import AppState, GestureState
from src.renderer.objects import HologramObject, build_object_catalogue
from src.ui.hud import HUD


class HologramRenderer:

    def __init__(self):
        cfg = CONFIG.renderer
        self._w   = cfg.window_width
        self._h   = cfg.window_height
        self._fov = cfg.fov
        self._near = cfg.near_clip
        self._far  = cfg.far_clip

        self._objects = None   # built after GL context is ready
        self._hud     = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Call once after pygame.init()."""
        pygame.display.set_mode(
            (self._w, self._h),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Gesture Hologram")

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.02, 0.02, 0.05, 1.0)

        # Build objects now that the GL context exists
        self._objects = build_object_catalogue()
        self._hud = HUD(self._w, self._h)

    def cleanup(self) -> None:
        if self._objects:
            for obj in self._objects:
                obj.cleanup()

    # ------------------------------------------------------------------
    # Per-frame render
    # ------------------------------------------------------------------

    def render(self, state: AppState, dt: float) -> None:
        """
        Render one frame.  dt = elapsed seconds since last frame.

        Auto-rotation is time-based:
            rot_y += speed_deg_per_sec * dt
        So it behaves identically at 30, 60, 120 fps.
        """
        # ── Update rotation ──────────────────────────────────────────────────
        if state.auto_rotate and state.gesture_state == GestureState.OPEN_PALM:
            state.rot_y += CONFIG.renderer.auto_rotate_speed * dt
            state.rot_y %= 360.0

        elif state.gesture_state == GestureState.TRACKING:
            # Map filtered index-tip position to rotation angles
            state.rot_x = state.hand.index_y * 360.0
            state.rot_y = state.hand.index_x * 360.0

        # ── OpenGL scene ─────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        gluPerspective(self._fov, self._w / self._h, self._near, self._far)
        glTranslatef(0.0, 0.0, state.zoom)

        glRotatef(state.rot_x, 1, 0, 0)
        glRotatef(state.rot_y, 0, 1, 0)

        # Draw current object
        obj = self._objects[state.obj_index]
        obj.render()

        # ── HUD overlay ──────────────────────────────────────────────────────
        self._hud.render(state, obj.name)

        pygame.display.flip()

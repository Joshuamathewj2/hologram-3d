"""
src/ui/hud.py
==============
Heads-up display rendered as an OpenGL pixel overlay.

Normal HUD (always visible):
  - Object name
  - Gesture hints
  - Current gesture state indicator
  - Swipe flash label

Debug overlay (F1 toggle):
  - FPS
  - Gesture state
  - Swipe velocity
  - Pinch distance
  - Hand confidence
  - Filtered vs raw index-tip coordinates
  - Hand visibility state
"""

import time
import pygame
from OpenGL.GL import *
from src.core.app_state import AppState, GestureState, HandVisibility


# State → colour mapping
_STATE_COLORS = {
    GestureState.IDLE:       (0.5, 0.5, 0.5),
    GestureState.TRACKING:   (0.0, 1.0, 1.0),
    GestureState.PINCHING:   (1.0, 0.5, 0.0),
    GestureState.SWIPING:    (0.0, 1.0, 0.0),
    GestureState.OPEN_PALM:  (1.0, 1.0, 0.0),
}

_VIS_COLORS = {
    HandVisibility.VISIBLE: (0.0, 1.0, 0.0),
    HandVisibility.GRACE:   (1.0, 1.0, 0.0),
    HandVisibility.LOST:    (1.0, 0.2, 0.2),
}


class HUD:

    def __init__(self, win_w: int, win_h: int):
        self._w = win_w
        self._h = win_h
        self._font_normal = None
        self._font_small  = None
        self._font_large  = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(self, state: AppState, obj_name: str) -> None:
        self._ensure_fonts()

        # ── Basic info ────────────────────────────────────────────────────────
        self._draw_text(20, self._h - 50,  f"Object : {obj_name}", (0, 255, 255))
        self._draw_text(20, self._h - 90,  "Move Hand = Rotate",   (180, 180, 180))
        self._draw_text(20, self._h - 125, "Pinch = Zoom",         (180, 180, 180))
        self._draw_text(20, self._h - 160, "Swipe = Switch Object",(180, 180, 180))
        self._draw_text(20, self._h - 195, "Open Palm = Auto-Rotate",(180,180,180))
        self._draw_text(20, self._h - 230, "F1 = Debug Overlay",   (100, 100, 100))

        # ── Gesture state badge ───────────────────────────────────────────────
        state_name  = state.gesture_state.name
        sc = _STATE_COLORS.get(state.gesture_state, (1,1,1))
        col = (int(sc[0]*255), int(sc[1]*255), int(sc[2]*255))
        self._draw_text(self._w - 220, self._h - 50, f"[ {state_name} ]", col)

        # ── Swipe flash ───────────────────────────────────────────────────────
        if time.time() < state.swipe_flash_until:
            arrow = ">>> SWIPE RIGHT >>>" if state.last_swipe_direction == "RIGHT" \
                    else "<<< SWIPE LEFT <<<"
            self._draw_large(self._w // 2 - 160, self._h // 2, arrow, (0, 255, 80))

        # ── Debug overlay ─────────────────────────────────────────────────────
        if state.debug_overlay:
            self._render_debug(state)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _render_debug(self, state: AppState) -> None:
        x = 20
        y = 380
        step = 28

        vis = state.hand_visibility
        vc = _VIS_COLORS.get(vis, (1,1,1))
        vis_col = (int(vc[0]*255), int(vc[1]*255), int(vc[2]*255))

        rows = [
            (f"FPS          : {state.fps:.1f}",         (200, 200, 200)),
            (f"Gesture      : {state.gesture_state.name}", (180, 240, 180)),
            (f"Hand Vis     : {vis.name}",               vis_col),
            (f"Swipe Vel    : {state.swipe_velocity:.3f}", (200, 200, 200)),
            (f"Pinch Dist   : {state.hand.pinch_dist:.3f}", (200, 200, 200)),
            (f"Pinch Active : {state.pinch_active}",     (200, 200, 200)),
            (f"Hand Conf    : {state.hand.hand_confidence:.2f}", (200, 200, 200)),
            (f"Filtered idx : ({state.hand.index_x:.3f}, {state.hand.index_y:.3f})", (100,220,255)),
            (f"Raw      idx : ({state.hand.raw_index_x:.3f}, {state.hand.raw_index_y:.3f})", (200,140,140)),
            (f"Object       : {state.obj_index}",        (200, 200, 200)),
        ]

        # Background panel
        self._draw_debug_bg(x - 5, y - 10, 300, len(rows) * step + 20)

        for i, (text, color) in enumerate(rows):
            self._draw_small(x, y + i * step, text, color)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_fonts(self):
        if self._font_normal is None:
            self._font_normal = pygame.font.SysFont("Consolas", 22)
            self._font_small  = pygame.font.SysFont("Consolas", 18)
            self._font_large  = pygame.font.SysFont("Consolas", 34, bold=True)

    def _draw_text(self, x: int, y: int, msg: str, color=(0,255,255)) -> None:
        self._blit(self._font_normal, x, y, msg, color)

    def _draw_small(self, x: int, y: int, msg: str, color=(200,200,200)) -> None:
        self._blit(self._font_small, x, y, msg, color)

    def _draw_large(self, x: int, y: int, msg: str, color=(0,255,80)) -> None:
        self._blit(self._font_large, x, y, msg, color)

    def _blit(self, font, x: int, y: int, msg: str, color) -> None:
        surface = font.render(msg, True, color)
        data    = pygame.image.tostring(surface, "RGBA", True)
        glWindowPos2d(x, y)
        glDrawPixels(
            surface.get_width(),
            surface.get_height(),
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            data,
        )

    def _draw_debug_bg(self, x, y, w, h) -> None:
        """Semi-transparent dark panel behind the debug text."""
        # Draw a filled quad using immediate mode (acceptable for a 2D overlay)
        glPushMatrix()
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self._w, 0, self._h, -1, 1)
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.0, 0.0, 0.0, 0.65)
        glBegin(GL_QUADS)
        glVertex2f(x,     y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x,     y + h)
        glEnd()
        glDisable(GL_BLEND)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

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

        # ── Gesture state badge (Always visible in top right) ──────────────────
        state_name = state.gesture_state.name
        # Only draw badge if it's an active gesture (optional, but requested "may remain")
        # I'll keep it as a gesture indicator for all states to show the live state.
        sc = _STATE_COLORS.get(state.gesture_state, (1, 1, 1))
        col = (int(sc[0]*255), int(sc[1]*255), int(sc[2]*255))
        self._draw_text(self._w - 220, self._h - 50, f"[ {state_name} ]", col)

        # ── Debug overlay (Toggled via F1) ────────────────────────────────────
        if state.debug_overlay:
            self._render_debug(state, obj_name)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _render_debug(self, state: AppState, obj_name: str) -> None:
        x = 20
        step = 28

        vis = state.hand_visibility
        vc = _VIS_COLORS.get(vis, (1,1,1))
        vis_col = (int(vc[0]*255), int(vc[1]*255), int(vc[2]*255))
        is_visible = vis in (HandVisibility.VISIBLE, HandVisibility.GRACE)
        
        frame_time_ms = (1000.0 / state.fps) if state.fps > 0 else 0.0

        rows = [
            (f"FPS          : {state.fps:.1f}",         (200, 200, 200)),
            (f"State        : {state.gesture_state.name}", (180, 240, 180)),
            (f"Object       : {obj_name}",               (200, 255, 200)),
            (f"Hand Visible : {is_visible}",             vis_col),
            (f"Swipe Vel    : {state.swipe_velocity:.3f}", (200, 200, 200)),
            (f"Pinch Dist   : {state.hand.pinch_dist:.3f}", (200, 200, 200)),
            (f"Confidence   : {state.hand.hand_confidence:.2f}", (200, 200, 200)),
            (f"Raw idx      : ({state.hand.raw_index_x:.3f}, {state.hand.raw_index_y:.3f})", (200,140,140)),
            (f"Filtered idx : ({state.hand.index_x:.3f}, {state.hand.index_y:.3f})", (100,220,255)),
            (f"Frame Time   : {frame_time_ms:.1f} ms",   (200, 200, 200)),
        ]

        # Draw panel from top-down
        start_y = self._h - 50
        
        # Background panel
        self._draw_debug_bg(x - 5, start_y - len(rows)*step - 5, 410, len(rows) * step + 20)

        for i, (text, color) in enumerate(rows):
            # Draw top-down. The top item is at (start_y - step), next is below it.
            self._draw_small(x, start_y - i * step, text, color)

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
        # tostring with "RGBA", True -> image is flipped vertically in raw buffer
        # This makes it natively upright for OpenGL which has origin at bottom-left
        data = pygame.image.tostring(surface, "RGBA", True)
        w, h = surface.get_width(), surface.get_height()
        
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self._w, 0, self._h, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(x, y, 0)
        
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        # Normal texture coordinates:
        # Texture bottom-left (0,0) -> Quad bottom-left (0,0)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        # Texture bottom-right (1,0) -> Quad bottom-right (w,0)
        glTexCoord2f(1, 0); glVertex2f(w, 0)
        # Texture top-right (1,1) -> Quad top-right (w,h)
        glTexCoord2f(1, 1); glVertex2f(w, h)
        # Texture top-left (0,1) -> Quad top-left (0,h)
        glTexCoord2f(0, 1); glVertex2f(0, h)
        glEnd()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        glBindTexture(GL_TEXTURE_2D, 0)
        glDeleteTextures(1, [tex])
        glPopAttrib()

    def _draw_debug_bg(self, x, y, w, h) -> None:
        """Semi-transparent dark panel behind the debug text."""
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self._w, 0, self._h, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glColor4f(0.0, 0.0, 0.0, 0.85)
        glBegin(GL_QUADS)
        glVertex2f(x,     y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x,     y + h)
        glEnd()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopAttrib()

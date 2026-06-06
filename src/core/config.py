"""
src/core/config.py
==================
Centralized, immutable configuration for the entire application.
All magic numbers live here — nowhere else.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class TrackingConfig:
    max_num_hands: int = 1
    min_detection_confidence: float = 0.70
    min_tracking_confidence: float = 0.70

    # EMA smoothing alpha (0 = max smooth, 1 = no smooth)
    ema_alpha: float = 0.25


@dataclass(frozen=True)
class SwipeConfig:
    # Minimum x-displacement (normalised 0-1) to count as a swipe
    displacement_threshold: float = 0.18
    # Minimum average velocity (units/sec) to count as a swipe
    velocity_threshold: float = 0.40
    # Gesture must complete within this time window (seconds)
    min_duration: float = 0.08
    max_duration: float = 0.80
    # Cooldown between accepted swipes
    cooldown: float = 1.20
    # History length (frames kept in deque)
    history_len: int = 30


@dataclass(frozen=True)
class PinchConfig:
    # Normalised distance below which pinch is active
    active_threshold: float = 0.06
    # Distance above which pinch is released (hysteresis)
    release_threshold: float = 0.10
    # Zoom mapping: pinch dist → zoom value
    zoom_min: float = -12.0
    zoom_max: float = -3.0
    pinch_dist_min: float = 0.02
    pinch_dist_max: float = 0.25


@dataclass(frozen=True)
class PalmConfig:
    # All 4 finger tips must be above their PIP joint
    confidence_threshold: float = 0.85


@dataclass(frozen=True)
class RendererConfig:
    window_width: int = 1400
    window_height: int = 900
    fov: float = 45.0
    near_clip: float = 0.1
    far_clip: float = 50.0
    auto_rotate_speed: float = 45.0      # degrees per second


@dataclass(frozen=True)
class HandLossConfig:
    # Seconds hand must be absent before state is frozen
    grace_period: float = 1.0
    # Seconds after grace before neutral pose recovery begins
    recovery_start: float = 1.0
    # Seconds to fully return to neutral pose (lerp duration)
    recovery_duration: float = 2.0
    # Seconds after which full state reset occurs
    reset_after: float = 3.0


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    swipe: SwipeConfig = field(default_factory=SwipeConfig)
    pinch: PinchConfig = field(default_factory=PinchConfig)
    palm: PalmConfig = field(default_factory=PalmConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    hand_loss: HandLossConfig = field(default_factory=HandLossConfig)


# Singleton — import this everywhere
CONFIG = AppConfig()

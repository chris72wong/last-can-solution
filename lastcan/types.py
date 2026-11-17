"""Shared immutable packets. All runtime timestamps use time.monotonic()."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Generic, TypeVar


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    captured_at: float
    image: Any  # full-resolution PIL RGB image


@dataclass(frozen=True)
class DetectionPacket:
    frame_id: int
    captured_at: float
    completed_at: float
    image: Any  # the EXACT PIL frame used for these boxes
    boxes: tuple[Detection, ...]
    simulated: bool = False
    error: str = ""


T = TypeVar("T")


class LatestValue(Generic[T]):
    """A single overwrite slot: slow consumers never build a frame backlog."""

    def __init__(self) -> None:
        self._value: T | None = None
        self._lock = Lock()

    def put(self, value: T | None) -> None:
        with self._lock:
            self._value = value

    def get(self) -> T | None:
        with self._lock:
            return self._value


DRIVE_ACTIONS = {
    "forward": (1, 0, 0), "back": (-1, 0, 0),
    "left": (0, 1, 0), "right": (0, -1, 0),
    "rotate_left": (0, 0, 1), "rotate_right": (0, 0, -1),
}


def drive_axes(actions: set[str], linear: float, angular: float) -> tuple[float, float, float]:
    axes = [0, 0, 0]
    for action in actions:
        if action in DRIVE_ACTIONS:
            for index, value in enumerate(DRIVE_ACTIONS[action]):
                axes[index] += value
    return (max(-1, min(1, axes[0])) * linear,
            max(-1, min(1, axes[1])) * linear,
            max(-1, min(1, axes[2])) * angular)


def fit_geometry(source: tuple[int, int], target: tuple[int, int]) -> tuple[float, float, float]:
    """Return scale and letterbox offsets; use the same transform for image/boxes."""
    sw, sh = source
    tw, th = target
    scale = min(max(1, tw) / max(1, sw), max(1, th) / max(1, sh))
    return scale, (tw - sw * scale) / 2, (th - sh * scale) / 2

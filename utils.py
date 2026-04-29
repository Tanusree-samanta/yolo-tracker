"""
utils.py
--------
Drawing helpers, colour palette, FPS counter, and object-counting logic.
All functions operate on OpenCV BGR frames (np.ndarray).
"""

from __future__ import annotations

import time
import colorsys
import numpy as np
import cv2


# -----------------------------------------------------------------------
# Colour palette
# -----------------------------------------------------------------------

def _generate_palette(n: int = 100) -> list[tuple[int, int, int]]:
    """
    Generate N visually distinct BGR colours using HSV colour space.
    Each track ID maps to a consistent colour.
    """
    palette = []
    for i in range(n):
        hue = i / n                    # evenly spaced hues
        saturation = 0.75 + 0.25 * ((i % 4) / 3)   # slight variation
        value = 0.85
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        bgr = (int(b * 255), int(g * 255), int(r * 255))
        palette.append(bgr)
    return palette


_PALETTE = _generate_palette(200)


def get_color(track_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR colour for a given track ID."""
    return _PALETTE[track_id % len(_PALETTE)]


# -----------------------------------------------------------------------
# Bounding box drawing
# -----------------------------------------------------------------------

def draw_track(
    frame: np.ndarray,
    track: dict,
    label: str,
    color: tuple[int, int, int] | None = None,
    box_thickness: int = 2,
    font_scale: float = 0.55,
    font_thickness: int = 1,
    corner_radius: int = 6,
) -> None:
    """
    Draw a single track onto *frame* (in-place).

    Parameters
    ----------
    frame          : BGR frame.
    track          : Dict with keys 'bbox', 'track_id', 'confidence', 'class_id'.
    label          : Human-readable label string, e.g. "person #3  87%".
    color          : BGR colour.  None → auto from track_id.
    box_thickness  : Line thickness in pixels.
    font_scale     : OpenCV font scale.
    font_thickness : OpenCV font thickness.
    corner_radius  : Radius for rounded-corner effect (0 = square corners).
    """
    x1, y1, x2, y2 = track["bbox"]
    tid = track["track_id"]

    if color is None:
        color = get_color(tid)

    # ── Bounding box ──────────────────────────────────────────────────
    if corner_radius > 0:
        _draw_rounded_rect(frame, x1, y1, x2, y2, color, box_thickness, corner_radius)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

    # ── Label background + text ───────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    pad = 4

    # Label background (above the box, clipped to frame top)
    lbl_y1 = max(y1 - text_h - baseline - pad * 2, 0)
    lbl_y2 = lbl_y1 + text_h + baseline + pad * 2
    lbl_x2 = min(x1 + text_w + pad * 2, frame.shape[1])

    cv2.rectangle(frame, (x1, lbl_y1), (lbl_x2, lbl_y2), color, cv2.FILLED)

    # Text in contrasting white/black
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

    cv2.putText(
        frame,
        label,
        (x1 + pad, lbl_y2 - baseline - pad),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )


def _draw_rounded_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    thickness: int,
    radius: int,
) -> None:
    """Draw a rectangle with slightly rounded corners."""
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    # Four straight edges (inset by radius)
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness)  # top
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness)  # bottom
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness)  # left
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness)  # right

    # Four arcs at corners
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0,  90,  color, thickness)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0,  90,  color, thickness)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r),  90, 0,  90,  color, thickness)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r),   0, 0,  90,  color, thickness)


# -----------------------------------------------------------------------
# Overlay helpers
# -----------------------------------------------------------------------

def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: tuple[int, int] = (10, 30),
    font_scale: float = 0.8,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Overlay the FPS counter on the frame (in-place)."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA,
    )


def draw_object_count(
    frame: np.ndarray,
    counts: dict[str, int],
    position: tuple[int, int] = (10, 65),
    font_scale: float = 0.55,
    line_height: int = 22,
) -> None:
    """
    Overlay per-class object counts on the frame (in-place).

    Parameters
    ----------
    counts   : {class_name: count}  e.g. {"person": 3, "car": 1}
    position : Top-left anchor for the first line.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position

    for i, (cls_name, count) in enumerate(counts.items()):
        text = f"{cls_name}: {count}"
        cy = y + i * line_height
        cv2.putText(frame, text, (x, cy), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, cy), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_info_bar(
    frame: np.ndarray,
    text: str,
    bar_height: int = 24,
    bg_color: tuple[int, int, int] = (30, 30, 30),
    text_color: tuple[int, int, int] = (200, 200, 200),
    font_scale: float = 0.45,
) -> None:
    """Draw a semi-transparent info bar at the very bottom of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), bg_color, cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame, text,
        (6, h - 7),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA,
    )


# -----------------------------------------------------------------------
# FPS counter
# -----------------------------------------------------------------------

class FPSCounter:
    """
    Smooth FPS counter using a rolling average over the last N frames.

    Usage::

        fps_counter = FPSCounter(window=30)
        fps_counter.tick()
        fps = fps_counter.fps   # float
    """

    def __init__(self, window: int = 30):
        self._window = window
        self._times: list[float] = []

    def tick(self) -> None:
        """Call once per processed frame."""
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def fps(self) -> float:
        """Current FPS estimate (returns 0 until at least 2 ticks)."""
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# -----------------------------------------------------------------------
# Object counter
# -----------------------------------------------------------------------

class ObjectCounter:
    """
    Counts unique object IDs seen per class across the current frame.

    A new instance of this class should be created each frame (or call
    `reset()` at the start of each frame) so the counts reflect only the
    *currently active* tracks.
    """

    def __init__(self):
        self._counts: dict[str, set[int]] = {}  # class_name → set of track IDs

    def update(self, tracks: list[dict], class_names: dict[int, str]) -> None:
        """
        Register the currently active tracks.

        Parameters
        ----------
        tracks       : List of track dicts from SORTTracker / DeepSORTTracker.
        class_names  : Mapping from class_id (int) → class_name (str).
        """
        self._counts.clear()
        for trk in tracks:
            name = class_names.get(trk["class_id"], f"cls_{trk['class_id']}")
            self._counts.setdefault(name, set()).add(trk["track_id"])

    @property
    def counts(self) -> dict[str, int]:
        """Return {class_name: count} for the last `update` call."""
        return {name: len(ids) for name, ids in self._counts.items()}


# -----------------------------------------------------------------------
# Format label string
# -----------------------------------------------------------------------

def format_label(class_name: str, track_id: int, confidence: float) -> str:
    """Build the label string shown next to each bounding box."""
    return f"{class_name} #{track_id}  {confidence:.0%}"


# -----------------------------------------------------------------------
# Video source helper
# -----------------------------------------------------------------------

def open_video_source(source: str | int) -> cv2.VideoCapture:
    """
    Open a video source (webcam index or file path).

    Parameters
    ----------
    source : 0, 1, 2, … for webcams  |  '/path/to/video.mp4' for files.

    Returns
    -------
    cv2.VideoCapture (already verified to be opened successfully).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video source: {source!r}\n"
            "  • For a webcam pass an integer index, e.g. 0\n"
            "  • For a file pass the full path, e.g. 'video.mp4'"
        )
    return cap

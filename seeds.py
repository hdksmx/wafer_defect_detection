"""
seeds.py
========

Generate *seed* line segments that will later be validated by the
_a contrario_ statistical test.

Two algorithms are supported:

1. **Probabilistic Hough Transform** (default)
   Fast and available in `skimage.transform`.

2. (optional) **LSD** ‒ Line Segment Detector (`pylsd2`)
   Useful for high‑resolution images with many texture edges.  Imported
   lazily so the dependency is only required when the method is used.

The returned list contains lightweight ``LineSegment`` objects with
pre‑computed geometry (end‑points, length, angle).  Coordinates follow
OpenCV / NumPy convention: **(x,y)** with *origin at top‑left*.

Example
-------
```python
mask = candidates.make_mask(img)
seeds = hough_seeds(mask,
                    angle_range=(80, 100),  # degrees
                    threshold=10,
                    min_len_px=int(img.shape[0] * 0.005))  # 0.5% height
```
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import List, Sequence, Tuple, Optional

import numpy as np
from skimage.transform import probabilistic_hough_line


__all__ = ["LineSegment", "hough_seeds", "lsd_seeds"]

# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------
@dataclass(slots=True)
class LineSegment:
    """Simple container for a 2‑D line segment."""
    p0: Tuple[int, int]  # (x0, y0)
    p1: Tuple[int, int]  # (x1, y1)
    angle_deg: float     # [0, 180)
    length: float        # Euclidean length

    @classmethod
    def from_endpoints(cls, p0, p1) -> "LineSegment":
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        length = hypot(dx, dy)
        angle = (degrees(atan2(-dy, dx)) + 180.0) % 180.0  # 0deg = +x
        return cls(p0=(int(x0), int(y0)),
                   p1=(int(x1), int(y1)),
                   angle_deg=angle,
                   length=length)

    def as_tuple(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return self.p0, self.p1


# ---------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------
def _angle_filter(seg: LineSegment,
                  angle_range: Optional[Tuple[float, float]]) -> bool:
    """Return True if segment angle is within *angle_range* or
    if angle_range is None (no filtering)."""
    if angle_range is None:
        return True
    lo, hi = angle_range
    if lo <= hi:
        return lo <= seg.angle_deg <= hi
    # wrap‑around (e.g. 170–10°)
    return seg.angle_deg >= lo or seg.angle_deg <= hi


def _post_filter(segments: Sequence[LineSegment],
                 *,
                 angle_range: Optional[Tuple[float, float]],
                 min_len_px: int) -> List[LineSegment]:
    """Apply angle & length constraints + deduplication."""
    out = []
    seen = set()
    for s in segments:
        if s.length < min_len_px:
            continue
        if not _angle_filter(s, angle_range):
            continue
        # deduplicate by near‑identical end‑points (within 1px)
        key = tuple(round(c, 0) for c in (*s.p0, *s.p1))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ---------------------------------------------------------------------
# 1) Hough
# ---------------------------------------------------------------------
def hough_seeds(mask: np.ndarray,
                *,
                angle_range: Optional[Tuple[float, float]] = None,
                angle_step: float = 0.5,
                threshold: int = 10,
                line_gap: int = 5,
                min_len_px: int = 20) -> List[LineSegment]:
    """
    Generate seed segments via Probabilistic Hough on *mask*.

    Parameters
    ----------
    mask : np.ndarray[bool] or uint8
        Binary mask from `candidates.make_mask`.
    angle_range : (float, float) or None, default=None
        Allowed segment angle in degrees. None → no restriction (0–180°).
    angle_step : float, default=0.5
        θ discretisation (deg) passed to Hough.
    threshold : int, default=10
        Minimum accumulator votes for a line.
    line_gap : int, default=5
        Maximum gap between pixels in the same line.
    min_len_px : int, default=20
        Minimum length accepted after detection.

    Returns
    -------
    List[LineSegment]
    """
    if mask.dtype != np.uint8:
        mask = (mask.astype(np.uint8)) * 255

    # Build θ array
    if angle_range is None:
        thetas = None  # skimage will scan full 0–π range
    else:
        lo, hi = angle_range
        thetas = np.deg2rad(np.arange(lo, hi + angle_step, angle_step))

    raw = probabilistic_hough_line(mask,
                                   threshold=threshold,
                                   line_length=min_len_px,
                                   line_gap=line_gap,
                                   theta=thetas)

    segs = [LineSegment.from_endpoints(*pair) for pair in raw]
    return _post_filter(segs,
                        angle_range=angle_range,
                        min_len_px=min_len_px)


# ---------------------------------------------------------------------
# 2) LSD (optional)
# ---------------------------------------------------------------------
def _require_lsd():
    try:
        import pylsd2 as lsd  # noqa: F401
    except ImportError as e:
        raise ImportError("pylsd2 not installed; install with `pip install pylsd2`") from e
    return lsd


def lsd_seeds(img: np.ndarray,
              *,
              angle_range: Optional[Tuple[float, float]] = None,
              min_len_px: int = 20) -> List[LineSegment]:
    """
    Generate seed segments using the LSD detector (`pylsd2`).

    Parameters
    ----------
    img : np.ndarray
        Original grayscale image (float32 or uint8).
    angle_range : (float, float) or None, default=None
        Allowed output angle (deg). None → no restriction.
    min_len_px : int, default=20
        Minimum accepted segment length.

    Returns
    -------
    List[LineSegment]
    """
    lsd = _require_lsd()

    # LSD expects uint8
    if img.dtype.kind == "f":
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)

    raw_segs = lsd.lsd(img_u8)
    segs = [LineSegment.from_endpoints((int(s[0]), int(s[1])),
                                       (int(s[2]), int(s[3]))) for s in raw_segs]
    return _post_filter(segs,
                        angle_range=angle_range,
                        min_len_px=min_len_px)
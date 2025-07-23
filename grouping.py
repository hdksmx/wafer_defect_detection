

"""
grouping.py
===========

Post‑processing helpers for _a contrario_ output.

After `acontrario.validate_segments` you obtain a list of
    (LineSegment, nfa)
sorted by NFA (ascending).  This module

1. **selects maximal meaningful segments**
   * keeps the lowest‑NFA segment when two overlap by ≥ `overlap_px`
2. **enforces the exclusion principle**
   * each image pixel is assigned to at most ONE accepted segment
3. (utility) rasterises accepted segments into a binary mask.

Public API
----------
* `select_segments(seg_nfa, image_shape, overlap_px=3)`
      -> List[(LineSegment, nfa)]
* `segments_to_mask(segments, shape, thick=1)` -> np.ndarray[bool]
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .seeds import LineSegment

__all__ = [
    "select_segments",
    "segments_to_mask",
]

# ---------------------------------------------------------------------
# Bresenham – duplicated to avoid circular import
# ---------------------------------------------------------------------
def _bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    """Generate integer pixel coords on the line segment."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


# ---------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------
def select_segments(
    seg_nfa: Sequence[Tuple[LineSegment, float]],
    image_shape: Tuple[int, int],
    *,
    overlap_px: int = 3,
) -> List[Tuple[LineSegment, float]]:
    """
    Apply maximality & exclusion to (segment, nfa) list.

    Parameters
    ----------
    seg_nfa : list[(LineSegment, float)]
        Output of `acontrario.validate_segments`, **MUST** be sorted by NFA↑.
    image_shape : (H, W)
        Shape of the original image so we can allocate an occupancy map.
    overlap_px : int, default=3
        If the number of shared pixels ≥ overlap_px the later segment is
        considered redundant and skipped.

    Returns
    -------
    accepted : list[(LineSegment, float)]
    """
    occ = np.zeros(image_shape, dtype=bool)
    accepted: List[Tuple[LineSegment, float]] = []

    for seg, nfa in seg_nfa:  # already sorted
        shared = 0
        pixels = list(_bresenham(*seg.p0, *seg.p1))
        for x, y in pixels:
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                if occ[y, x]:
                    shared += 1
                    if shared >= overlap_px:
                        break
        if shared >= overlap_px:
            continue  # skip redundant
        # accept
        accepted.append((seg, nfa))
        for x, y in pixels:
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                occ[y, x] = True
    return accepted


# ---------------------------------------------------------------------
# Mask rasterisation
# ---------------------------------------------------------------------
def segments_to_mask(
    segments: Sequence[LineSegment],
    shape: Tuple[int, int],
    *,
    thick: int = 1,
) -> np.ndarray:
    """
    Rasterise *segments* into a boolean mask.

    Parameters
    ----------
    segments : iterable[LineSegment]
    shape : (H, W)
    thick : int, default=1
        Thickness (dilation) of the rasterised line.

    Returns
    -------
    mask : np.ndarray[bool], shape=shape
    """
    mask = np.zeros(shape, dtype=bool)
    for seg in segments:
        for x, y in _bresenham(*seg.p0, *seg.p1):
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                mask[y, x] = True

    if thick > 1:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=thick - 1)
    return mask
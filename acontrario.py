

"""
acontrario.py
=============

Statistical validation of *seed* line segments using the _a contrario_
framework described by Desolneux et al.  Every candidate segment returned
by `seeds.hough_seeds`/`lsd_seeds` is assigned a **Number of False
Alarms (NFA)**; only those with ``NFA ≤ eps`` are kept.

Implementation notes
--------------------
* Two variants are provided

    * **Binomial tail** – assumes a *global* detection probability p.
    * **Hoeffding bound** – uses a *local* probability map p(x,y) and the
      Hoeffding (1963) inequality to keep computation O(1) per segment.

* A pixel contributes to the inlier count **k** if **all** are true
    1. It belongs to the candidate mask I_B (from `candidates.make_mask`).
    2. Its gradient orientation θ(x,y) is within ±`angle_tol` of the
       *perpendicular* of the segment direction (cf. scratch definition).

* “Aligned” means gradient ⟂ segment, because scratches are darker
  *lines* orthogonal to the intensity gradient.

Public API
----------
`validate_segments(segments, theta_map, mask, eps=1.0, **kwargs)
     -> List[Tuple[LineSegment, float]]`
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.stats import binom

from .seeds import LineSegment

__all__ = [
    "nfa_binomial",
    "nfa_hoeffding",
    "validate_segments",
]

# ----------------------------------------------------------------------
# NFA helpers
# ----------------------------------------------------------------------
def binom_tail(n: int, k: int, p: float) -> float:
    """P(S_n ≥ k) for S_n ~ Binom(n,p) (survival function)."""
    if k <= 0:
        return 1.0
    # sf(k-1) because sf is P(X > k-1)
    return float(binom.sf(k - 1, n, p))


def nfa_binomial(n: int, k: int, p: float, n_tests: int) -> float:
    """NFA using exact binomial tail."""
    return n_tests * binom_tail(n, k, p)


def hoeffding_tail(n: int, k: int, p_bar: float) -> float:
    """
    Hoeffding inequality upper bound for P(S_n ≥ k).

    P(S_n ≥ k) ≤ exp(-2 n (k/n - p̄)^2)  for Bernoulli variables in [0,1].
    """
    if k <= p_bar * n:
        return 1.0
    diff = k / n - p_bar
    return math.exp(-2.0 * n * diff * diff)


def nfa_hoeffding(n: int, k: int, p_bar: float, n_tests: int) -> float:
    """NFA using Hoeffding bound."""
    return n_tests * hoeffding_tail(n, k, p_bar)


# ----------------------------------------------------------------------
# Pixel sampling util
# ----------------------------------------------------------------------
def _bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    """Generate pixel coordinates along the segment (inclusive)."""
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


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
@dataclass(slots=True)
class _Params:
    eps: float
    angle_tol: float
    window: int
    use_hoeffding: bool


def _local_probability_map(mask: np.ndarray, window: int) -> np.ndarray:
    """
    Sliding-window mean density map.

    window – half window size (square side = 2*window+1).
    """
    from scipy.ndimage import uniform_filter

    if window <= 0:
        # Global average
        return mask.mean() * np.ones_like(mask, dtype=np.float32)

    size = 2 * window + 1
    p = uniform_filter(mask.astype(np.float32), size=size, mode="nearest")
    return p


def _aligned_count(
    seg: LineSegment,
    theta: np.ndarray,
    mask: np.ndarray,
    p_map: np.ndarray,
    angle_tol: float,
) -> Tuple[int, float]:
    """
    Return (k, p_bar) for the pixels of *seg* that satisfy mask==True.
    p_bar is the mean p_i over those pixels (for Hoeffding).
    """
    aligned = 0
    p_accum = 0.0
    n = 0
    # Segment direction vector
    seg_angle = math.radians(seg.angle_deg)
    # Scratch line is perpendicular → gradient should align with seg_angle ±π/2
    target = (seg_angle + math.pi / 2) % math.pi

    for x, y in _bresenham(*seg.p0, *seg.p1):
        if y < 0 or y >= mask.shape[0] or x < 0 or x >= mask.shape[1]:
            continue
        if not mask[y, x]:
            continue
        n += 1
        p_accum += float(p_map[y, x])
        # unsigned angle difference
        diff = abs(theta[y, x] - target)
        diff = min(diff, math.pi - diff)
        if diff < angle_tol:
            aligned += 1

    if n == 0:
        return 0, 0.0
    return aligned, p_accum / n


def validate_segments(
    segments: List[LineSegment],
    theta_map: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    eps: float = 1.0,
    angle_tol: float = math.radians(1.0),
    local_window: int = 15,
    n_tests: int | None = None,
    use_hoeffding: bool = True,
) -> List[Tuple[LineSegment, float]]:
    """
    Filter *segments* returning those with NFA ≤ eps.

    Parameters
    ----------
    segments : list[LineSegment]
    theta_map : np.ndarray
        Orientation map (output of orientation.sobel_orientation).
    candidate_mask : np.ndarray[bool]
        Mask I_B from candidates.make_mask.
    eps : float, default=1
        Meaningfulness threshold (accepted if NFA ≤ eps).
    angle_tol : float, default=1° (radians)
        Tolerance when comparing gradient dir to segment ⟂ .
    local_window : int, default=15
        Half window size for probability map. 0 → global p only.
    n_tests : int | None
        If None, defaults to (image_height * image_width * 40)
        assuming ~0.5° angle discretisation over ±10°.
    use_hoeffding : bool, default=True
        If False, always use global Binomial tail.

    Returns
    -------
    list of (segment, NFA)
    """
    h, w = candidate_mask.shape
    if n_tests is None:
        n_tests = h * w * 40  # crude upper bound

    p_map = _local_probability_map(candidate_mask, local_window)

    out: List[Tuple[LineSegment, float]] = []
    for seg in segments:
        k, p_bar = _aligned_count(
            seg, theta_map, candidate_mask, p_map, angle_tol
        )
        if k == 0:
            continue
        n = int(round(seg.length))
        if n == 0:
            continue

        if use_hoeffding and local_window > 0:
            nfa = nfa_hoeffding(n, k, p_bar, n_tests)
        else:
            # global p = overall density
            p_global = candidate_mask.mean()
            nfa = nfa_binomial(n, k, p_global, n_tests)

        if nfa <= eps:
            out.append((seg, nfa))

    # sort by NFA ascending
    out.sort(key=lambda t: t[1])
    return out
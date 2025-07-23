"""
orientation.py
==============

Compute gradient magnitude and **orientation** (angle) for a grayscale
image.  These maps are a core input for the _a contrario_ validation step
of the wafer‑scratch detection pipeline.

All functions expect **float32** input normalised to [0,1] and return
float32 outputs.

Angle Convention
----------------
The returned orientation is the **gradient direction** θ ∈ [0, π):

    θ = 0      → gradient points to the +x (right) direction
    θ = π/2    → gradient points up (–y in image coords)
    θ = π      → gradient points to –x (left)

Since θ and θ+π represent the same direction, collapsing to [0,π) is
convenient for “unsigned” comparisons:

    abs(θ₁−θ₂) < tol     # correctly handles wrap‑around at π.

Public API
----------
* `sobel(img, ksize=3) -> (gx, gy)`
* `sobel_orientation(img, ksize=3, *, return_magnitude=False)`
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Union

__all__ = [
    "sobel",
    "sobel_orientation",
]

# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def _check_float32(img: np.ndarray) -> None:
    if img.dtype.kind != "f":
        raise TypeError("Input image must be float32 (range [0,1])")


def sobel(img: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute x and y Sobel derivatives.

    Parameters
    ----------
    img : np.ndarray
        Grayscale float32 image in [0,1].
    ksize : int, default=3
        Sobel kernel size (1, 3, 5, or 7).

    Returns
    -------
    gx, gy : np.ndarray
        Float32 derivative images, same shape as *img*.
    """
    _check_float32(img)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize, borderType=cv2.BORDER_REPLICATE)
    return gx, gy


def sobel_orientation(
    img: np.ndarray,
    ksize: int = 3,
    *,
    return_magnitude: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute gradient **orientation** (and optionally magnitude).

    Parameters
    ----------
    img : np.ndarray
        Grayscale float32 image in [0,1].
    ksize : int, default=3
        Sobel kernel size.
    return_magnitude : bool, default=False
        If True, also return the gradient magnitude.

    Returns
    -------
    theta : np.ndarray, float32
        Orientation map in **radians**, range [0, π).
    (optionally) mag : np.ndarray, float32
        Gradient magnitude (sqrt(gx²+gy²)).
    """
    gx, gy = sobel(img, ksize=ksize)

    # Angle in [-π, π]; shift to [0, 2π) then mod π → [0, π)
    theta = (np.arctan2(gy, gx) + np.pi) % np.pi
    theta = theta.astype(np.float32)

    if return_magnitude:
        mag = np.hypot(gx, gy).astype(np.float32)
        return theta, mag
    return theta

"""
candidates.py
=============

Pixel‑level candidate mask for wafer‑scratch detection.

Implements the two conditions described in the _a contrario_ paper
(Newson et al.):

    c₁(x,y) : |I_g(x,y) − I_m(x,y)| ≥ s_med
    c₂(x,y) : |L₃(x,y) − R₃(x,y)| ≤ s_avg

where
* I_g  – Gaussian‑blurred image (σ ≈ 1px)
* I_m  – **horizontal** median over 5pixels
* L₃   – mean of the 3pixels immediately LEFT  of (x,y)
* R₃   – mean of the 3pixels immediately RIGHT of (x,y)

A pixel is marked *True* in the binary mask `I_B` if **both** conditions
hold.  The thresholds `s_med` and `s_avg` are expressed using the same
intensity scale as the input image (float32, [0,1]).

Public API
----------
`make_mask(img, s_med=3/255, s_avg=20/255,…) -> np.ndarray[bool]`
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d, shift

__all__ = ["make_mask"]

# ---------------------------------------------------------------------
# Helper kernels
# ---------------------------------------------------------------------
def _left_right_means(img: np.ndarray, width: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean of *width* pixels immediately to the left / right of each pixel.

    For example, for width=3 and pixel (x,y) the *left* mean is
    the average of {x-3, x-2, x-1} while the *right* mean is the
    average of {x+1, x+2, x+3}.

    Implementation uses a centred uniform filter followed by an integer
    roll (shift) so we never rely on ``origin`` values that exceed the
    SciPy limit.

    Parameters
    ----------
    img : np.ndarray, float32 in [0,1]
    width : int, default=3
        Number of neighbouring pixels to average (must be ≥1).

    Returns
    -------
    left_mean, right_mean : np.ndarray – same shape as *img*
    """
    if width < 1:
        raise ValueError("width must be ≥ 1")

    # Centred moving‑average
    centred = uniform_filter1d(img, size=width, axis=1, mode="nearest")

    half = (width + 1) // 2  # integer shift
    # left: shift kernel centre to the RIGHT  → looks backwards
    left_mean = np.roll(centred, +half, axis=1)
    # right: shift centre to the LEFT → looks forwards
    right_mean = np.roll(centred, -half, axis=1)

    return left_mean, right_mean


def make_mask(
    img: np.ndarray,
    *,
    s_med: float = 3.0 / 255.0,
    s_avg: float = 20.0 / 255.0,
    gauss_sigma: float = 1.0,
    median_width: int = 5,
    lr_width: int = 3,
) -> np.ndarray:
    """
    Generate binary candidate mask *I_B* according to conditions (c₁, c₂).

    Parameters
    ----------
    img : np.ndarray
        Grayscale float32 image in [0,1].
    s_med : float, default=3/255
        Threshold for |I_g − I_m|.
    s_avg : float, default=20/255
        Threshold for |mean_left − mean_right|.
    gauss_sigma : float, default=1.0
        σ for Gaussian blur when computing I_g.
    median_width : int, default=5
        Width of horizontal median filter (odd).
    lr_width : int, default=3
        Width for left/right mean windows.

    Returns
    -------
    mask : np.ndarray, bool
        True where both conditions satisfied.
    """
    if img.dtype.kind != "f":
        raise TypeError("Input image must be float32 in [0,1]")

    # --- I_g : Gaussian blur ------------------------------------------------
    I_g = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=gauss_sigma, borderType=cv2.BORDER_REPLICATE)

    # --- I_m : horizontal median -------------------------------------------
    if median_width % 2 == 0:
        raise ValueError("median_width must be odd")
    I_m = median_filter(img, size=(1, median_width), mode="nearest")

    # --- left / right means -------------------------------------------------
    L3, R3 = _left_right_means(img, width=lr_width)

    # --- Conditions ---------------------------------------------------------
    c1 = np.abs(I_g - I_m) >= s_med
    c2 = np.abs(L3 - R3) <= s_avg
    mask = np.logical_and(c1, c2)

    return mask
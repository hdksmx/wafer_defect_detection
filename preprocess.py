

"""
preprocess.py
=============

Basic image–preprocessing utilities used in the wafer‑scratch detection
pipeline.

Functions
---------
* **gaussian**   – small Gaussian blur (noise suppression).
* **clahe**      – Contrast Limited Adaptive Histogram Equalisation.
* **preprocess** – convenience wrapper that selects one of the above
                   depending on CLI / caller arguments.

All routines accept and return **float32** arrays in the range [0,1].
I/O conversion (uint8 ↔ float) is handled in ``io_utils/io.py``.
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple, Literal

__all__ = [
    "gaussian",
    "clahe",
    "preprocess",
]


# ----------------------------------------------------------------------
# Individual steps
# ----------------------------------------------------------------------
def gaussian(img: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian‑blur *img*.

    Parameters
    ----------
    img : np.ndarray
        Grayscale float32 image in [0,1].
    ksize : int, default=3
        Kernel size (odd).
    sigma : float, default=1.0
        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray – blurred image, float32
    """
    if img.dtype.kind != "f":
        raise TypeError("Input image must be float32 in [0,1]")
    if ksize % 2 != 1:
        raise ValueError("ksize must be odd")

    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)


def clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited AHE) to *img*.

    Parameters
    ----------
    img : np.ndarray
        Grayscale float32 image in [0,1].
    clip_limit : float, default=2.0
        Threshold for contrast limiting.
    tile_grid_size : (int, int), default=(8, 8)
        Size of the grid for histogram equalisation.

    Returns
    -------
    np.ndarray – contrast‑enhanced float32 image in [0,1]
    """
    if img.dtype.kind != "f":
        raise TypeError("Input image must be float32 in [0,1]")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # cv2 CLAHE expects 8‑bit, so convert temporarily
    tmp = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    out = clahe.apply(tmp)
    return out.astype(np.float32) / 255.0


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
def preprocess(
    img: np.ndarray,
    method: Literal["gaussian", "clahe", "gauss_clahe", "none"] = "gaussian",
    *,
    # Gaussian params
    ksize: int = 3,
    sigma: float = 1.0,
    # CLAHE params
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Run selected preprocessing *method*.

    Available methods
    -----------------
    * 'gaussian'     – Gaussian blur only
    * 'clahe'        – CLAHE only
    * 'gauss_clahe'  – Gaussian → CLAHE
    * 'none'         – return input unchanged

    Extra keyword arguments are forwarded to the respective functions.

    Notes
    -----
    All outputs stay float32 in the range [0,1].
    """
    if method == "none":
        return img.copy()
    elif method == "gaussian":
        return gaussian(img, ksize=ksize, sigma=sigma)
    elif method == "clahe":
        return clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    elif method == "gauss_clahe":
        g = gaussian(img, ksize=ksize, sigma=sigma)
        return clahe(g, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    else:
        raise ValueError(f"Unknown preprocess method: {method}")
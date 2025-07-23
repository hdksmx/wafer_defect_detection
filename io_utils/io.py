

"""
io_utils/io.py
==============

Light‑weight image I/O helpers for the wafer‑scratch detection project.

Responsibilities
----------------
* Load a grayscale image as **float32** (optionally normalised to 0‑1).
* Save an image (mask or debug view) making sure the parent directory exists.
* One‑liner helper to drop intermediates into ``results/debug_img`` using
  `path.debug_img_path()`.

No heavy lifting (CLI parsing, logging, timing) lives here; those belong to
upper‑level modules.
"""
from __future__ import annotations

import os
from typing import Union

import cv2
import numpy as np

from .path import ensure_dir, debug_img_path

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    "load_gray",
    "save_image",
    "save_debug",
]

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------
def load_gray(src: PathLike, normalise: bool = True) -> np.ndarray:
    """
    Read *src* as **grayscale** float32 array.

    Parameters
    ----------
    src : str | os.PathLike
        Image path.
    normalise : bool, default=True
        If True, divide by 255 so the output is in [0, 1].

    Returns
    -------
    img : np.ndarray, float32
    """
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {src}")

    img = img.astype(np.float32)
    if normalise:
        img /= 255.0
    return img


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
def _to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert *img* to uint8, scaling if it is float.

    * If the dtype is uint8 already, return as‑is (copy).
    * If float → assume range [0,1] or [0,255]; clip and scale accordingly.
    """
    if img.dtype == np.uint8:
        return img.copy()
    if img.dtype.kind in {"f"}:
        if img.max() <= 1.0:
            img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def save_image(dst: PathLike, img: np.ndarray, *, auto_mkdir: bool = True) -> None:
    """
    Save *img* to *dst* (PNG/JPEG). Creates parent directories if needed.

    Parameters
    ----------
    dst : str | os.PathLike
        Destination file path.
    img : np.ndarray
        Image to write.
    auto_mkdir : bool, default=True
        If True (default) create parent directories automatically.
    """
    if auto_mkdir:
        ensure_dir(os.path.dirname(str(dst)))
    cv2.imwrite(str(dst), _to_uint8(img))


def save_debug(img: np.ndarray, filename: str | None = None) -> str:
    """
    Convenience wrapper: dump *img* into ``results/debug_img``.

    Returns the absolute path where the file was written.
    """
    path = debug_img_path(filename)
    save_image(path, img, auto_mkdir=False)  # dir already ensured
    return path
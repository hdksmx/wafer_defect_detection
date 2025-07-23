"""
post.py
=======

Post‑processing and visualisation helpers for the wafer‑scratch detection
pipeline.

Typical workflow
----------------
```
mask      = grouping.segments_to_mask(segments, img.shape)
thin      = skeletonise(mask)           # 1‑pixel centre‑line
vis_thin  = overlay(img, thin, color=(0, 255, 0), alpha=0.7)
thick     = thicken(thin, 3)            # for UI visibility
vis_thick = overlay(img, thick, color=(0, 0, 255), alpha=0.5)
```

Functions
---------
* **skeletonise(mask)** – centre‑line extraction.
* **thicken(mask, px)** – binary dilation by *px* pixels.
* **overlay(img, mask, color, alpha)** – RGBA compositing.

All inputs accept either *float32 [0,1]* or *uint8 [0,255]* images.
All returned images match the dtype/range of the *img* argument.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

__all__ = [
    "skeletonise",
    "thicken",
    "overlay",
    "draw_segments",
]


# ---------------------------------------------------------------------
# Binary mask ops
# ---------------------------------------------------------------------
def skeletonise(mask: np.ndarray) -> np.ndarray:
    """
    Convert *mask* to a 1‑pixel‑wide skeleton.  Uses
    `skimage.morphology.skeletonize`.

    Parameters
    ----------
    mask : np.ndarray[bool] | uint8
        Binary scratch mask.

    Returns
    -------
    np.ndarray[bool] – skeleton mask.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return skeletonize(mask)


def thicken(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    """
    Dilate *mask* by *(pixels‑1)* iterations using a 3×3 square SE.

    Parameters
    ----------
    mask : np.ndarray[bool] | uint8
    pixels : int, default=3
        Resulting half‑thickness. `pixels=1` → unchanged.

    Returns
    -------
    np.ndarray[bool]
    """
    if pixels <= 1:
        return mask.astype(bool)
    iterations = pixels - 1
    dilated = binary_dilation(mask.astype(bool), iterations=iterations)
    return dilated


# ---------------------------------------------------------------------
# Segment visualisation
# ---------------------------------------------------------------------
def draw_segments(
    img: np.ndarray,
    segments,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """
    Return an RGB/BGR image with line *segments* rendered on top.

    Parameters
    ----------
    img : np.ndarray
        Source image (grayscale or RGB, uint8 [0,255] or float [0,1]).
    segments : Iterable[LineSegment] | list[tuple[(x0,y0),(x1,y1)]]
        Segment objects from seeds.py or simple endpoint tuples.
    color : tuple, default=(255,0,0)
        BGR colour to draw.
    thickness : int, default=1
        Line thickness in pixels.

    Returns
    -------
    np.ndarray – same dtype / scale as *img* with segments overlaid.
    """
    # Ensure 3‑channel BGR uint8 base image
    if img.ndim == 2:
        base = cv2.cvtColor(
            (img * 255).astype(np.uint8) if img.dtype.kind == "f" else img,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        base = (
            (img * 255).round().astype(np.uint8)
            if img.dtype.kind == "f"
            else img.copy()
        )

    # Robustly iterate endpoints
    for seg in segments:
        try:
            p0, p1 = seg.p0, seg.p1  # LineSegment dataclass
        except AttributeError:
            p0, p1 = seg  # plain tuple
        cv2.line(base, p0, p1, color, thickness, cv2.LINE_AA)

    # Convert back to original dtype/scale
    if img.dtype.kind == "f":
        base = base.astype(np.float32) / 255.0
    return base

# ---------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------
def _prepare_float(img: np.ndarray) -> np.ndarray:
    """Return float32 RGB image in [0,1]."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype.kind != "f":
        img = img.astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def overlay(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Alpha‑blend *mask* on top of *img*.

    Parameters
    ----------
    img : np.ndarray
        Grayscale or RGB image, uint8 [0,255] or float32 [0,1].
    mask : np.ndarray[bool] | uint8
        Binary mask.
    color : 3‑tuple, default=(0,255,0)
        Overlay colour (BGR).
    alpha : float, default=0.6
        Opacity of the mask colour.

    Returns
    -------
    out : np.ndarray – same dtype/scale as *img*.
    """
    base = _prepare_float(img)
    if mask.dtype != bool:
        mask_bool = mask.astype(bool)
    else:
        mask_bool = mask

    colour = np.array(color, dtype=np.float32) / 255.0
    overlay_img = base.copy()
    overlay_img[mask_bool] = (
        alpha * colour + (1.0 - alpha) * base[mask_bool]
    )

    # Return image in original dtype/range
    if img.dtype == np.uint8:
        overlay_img = (overlay_img * 255.0).round().astype(np.uint8)
    return overlay_img

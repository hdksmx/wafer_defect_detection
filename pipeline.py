"""
pipeline.py
===========

End‑to‑end orchestration of the wafer‑scratch detection pipeline.

Usage (stand‑alone)
-------------------
```bash
python -m wafer_defect_detection.pipeline \
    --img input/wafer.png \
    --out results/mask.png \
    --preprocess gauss_clahe \
    --s_med 0.01 --s_avg 0.08 \
    --eps 1.0
```

The script prints per‑stage timings and writes:
* `--out`          : final binary mask (uint8 0/255)
* `--overlay` (opt): colour overlay PNG for visual inspection
* intermediate debug images in `results/debug_img/` (when `--debug`)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import datetime
import contextlib
from io import StringIO

import cv2  # used for drawing segments when debugging

# -------------------------------------------------------------------
# Robust imports: support running either as package *or* as script
# (i.e. `python pipeline.py` from inside the wafer_defect_detection
#  folder).  If relative import fails, we append the parent directory
#  to sys.path then fall back to absolute package imports.
# -------------------------------------------------------------------
try:
    # normal package context  → works when you do
    #   python -m wafer_defect_detection.pipeline ...
    from wafer_defect_detection.io_utils import path
    from wafer_defect_detection.io_utils.io import load_gray, save_image
    from wafer_defect_detection.io_utils.timing import Timer
    from wafer_defect_detection import (
        preprocess,
        candidates,
        orientation,
        seeds,
        acontrario,
        grouping,
        post,
    )
except ImportError:  # running as standalone script
    import os  # noqa: E402

    script_dir = Path(__file__).resolve().parent
    sys.path.append(str(script_dir.parent))  # add project root to PYTHONPATH

    from wafer_defect_detection.io_utils import path  # type: ignore
    from wafer_defect_detection.io_utils.io import load_gray, save_image  # type: ignore
    from wafer_defect_detection.io_utils.timing import Timer  # type: ignore
    from wafer_defect_detection import (  # type: ignore
        preprocess,
        candidates,
        orientation,
        seeds,
        acontrario,
        grouping,
        post,
    )


# ----------------------------------------------------------------------
# Core pipeline
# ----------------------------------------------------------------------
def run(
    img_path: Path,
    *,
    preprocess_method: str = "clahe",
    gauss_ksize: int = 3,
    gauss_sigma: float = 1.0,
    clahe_clip: float = 2.0,
    clahe_tiles: int = 8,
    s_med: float = 3.0 / 255.0,
    s_avg: float = 20.0 / 255.0,
    eps: float = 1.0,
    angle_tol_deg: float = 1.0,
    local_window: int = 15,
    debug: bool = False,
) -> np.ndarray:
    """
    Execute full detection pipeline on *img_path*.

    Returns
    -------
    mask : np.ndarray[bool] – final scratch mask.
    """
    # ------------------------------------------------------------------
    # 0. Load
    # ------------------------------------------------------------------
    with Timer("load"):
        img = load_gray(img_path)

    # ------------------------------------------------------------------
    # 1. Pre‑process
    # ------------------------------------------------------------------
    with Timer("preprocess"):
        img_p = preprocess.preprocess(
            img,
            method=preprocess_method,
            ksize=gauss_ksize,
            sigma=gauss_sigma,
            clip_limit=clahe_clip,
            tile_grid_size=(clahe_tiles, clahe_tiles),
        )
    if debug:
        save_image(path.debug_img_path(f"01_pre_{img_path.stem}.png"), img_p)

    # ------------------------------------------------------------------
    # 2. Candidate mask
    # ------------------------------------------------------------------
    with Timer("candidates"):
        mask_cand = candidates.make_mask(
            img_p, s_med=s_med, s_avg=s_avg, gauss_sigma=gauss_sigma
        )
    if debug:
        save_image(path.debug_img_path(f"02_candidates_{img_path.stem}.png"), mask_cand.astype(np.uint8) * 255)

    # ------------------------------------------------------------------
    # 3. Gradient orientation
    # ------------------------------------------------------------------
    with Timer("orientation"):
        theta_map = orientation.sobel_orientation(img_p, return_magnitude=False)
    if debug:
        # visualise θ map as grayscale (0°→0, 180°→255)
        theta_vis = (theta_map / np.pi * 255.0).astype(np.uint8)
        save_image(path.debug_img_path(f"03_theta_{img_path.stem}.png"), theta_vis)

    # ------------------------------------------------------------------
    # 4. Seed generation
    # ------------------------------------------------------------------
    with Timer("seeds"):
        seed_segments = seeds.hough_seeds(
            mask_cand,
            min_len_px=max(20, int(img.shape[0] * 0.005)),
        )
    if debug:
        # draw raw seed segments in blue
        vis_seeds = post.draw_segments(img, seed_segments, color=(255, 0, 0))
        save_image(path.debug_img_path(f"04_seeds_{img_path.stem}.png"), vis_seeds)

    # ------------------------------------------------------------------
    # 5. A‑contrario validation
    # ------------------------------------------------------------------
    with Timer("acontrario"):
        seg_nfa = acontrario.validate_segments(
            seed_segments,
            theta_map,
            mask_cand,
            eps=eps,
            angle_tol=np.deg2rad(angle_tol_deg),
            local_window=local_window,
        )
    if debug:
        vis_valid = post.draw_segments(img, [s for s, _ in seg_nfa], color=(0, 255, 0))
        save_image(path.debug_img_path(f"05_validated_{img_path.stem}.png"), vis_valid)

    # ------------------------------------------------------------------
    # 6. Maximality & exclusion
    # ------------------------------------------------------------------
    with Timer("grouping"):
        accepted = grouping.select_segments(seg_nfa, img.shape)

    # ------------------------------------------------------------------
    # 7. Raster mask
    # ------------------------------------------------------------------
    with Timer("rasterise"):
        mask_final = grouping.segments_to_mask(
            [s for s, _ in accepted], img.shape, thick=1
        )
    if debug:
        save_image(path.debug_img_path(f"06_mask_{img_path.stem}.png"), mask_final.astype(np.uint8) * 255)

    if debug:
        overlay = post.overlay(img, mask_final, color=(0, 0, 255), alpha=0.6)
        save_image(path.debug_img_path(f"99_overlay_{img_path.stem}.png"), overlay)

    return mask_final


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wafer scratch detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--img", type=Path, required=True, help="Input image file")
    p.add_argument("--out", type=Path, required=True, help="Output mask PNG")
    p.add_argument("--overlay", type=Path, help="Optional colour overlay PNG")
    p.add_argument("--preprocess", choices=["gaussian", "clahe", "gauss_clahe", "none"], default="clahe")
    p.add_argument("--s_med", type=float, default=3.0 / 255.0)
    p.add_argument("--s_avg", type=float, default=20.0 / 255.0)
    p.add_argument("--eps", type=float, default=1.0)
    p.add_argument("--debug", action="store_true", help="Save intermediate images")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    start_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mask = run(args.img, preprocess_method=args.preprocess, s_med=args.s_med, s_avg=args.s_avg, eps=args.eps,
               debug=args.debug)

    mask_path = path.run_path(Path(args.out).name)
    save_image(mask_path, mask.astype(np.uint8) * 255)

    if args.overlay:
        img = load_gray(args.img, normalise=False)
        overlay = post.overlay(img, mask, color=(0, 0, 255), alpha=0.5)
        overlay_path = path.run_path(Path(args.overlay).name)
        save_image(overlay_path, overlay)

    # ---- write run log -------------------------------------------------
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        Timer.summary()
    timer_txt = buf.getvalue()

    log_txt = (
        f"run_start : {start_ts}\n"
        f"input_img : {args.img}\n"
        f"preprocess: {args.preprocess}\n"
        f"s_med     : {args.s_med}\n"
        f"s_avg     : {args.s_avg}\n"
        f"eps       : {args.eps}\n"
        f"mask_path : {mask_path}\n"
        f"overlay   : {overlay_path if args.overlay else 'N/A'}\n"
        f"---------- timings ----------\n"
        f"{timer_txt}"
    )
    with open(path.run_log_path("run.txt"), "w") as f:
        f.write(log_txt)


if __name__ == "__main__":
    main()
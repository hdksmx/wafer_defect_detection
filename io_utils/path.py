"""
io_utils/path.py
================
Utility helpers for consistent file‑path handling across the wafer
defect‑detection project.

Conventions
-----------
* The *project root* is assumed to be two levels above this file.
  (.. / ..)
* All run‑time artefacts live under a single top‑level directory
  called ``results/`` relative to the project root, unless the
  environment variable ``WDD_OUTDIR`` overrides it.

Directory layout
----------------
result/
├── debug_img/          # intermediate visualisations
└── csv/                # timing logs, metrics, etc.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Sequence

def ensure_dir(path: str) -> None:
    """Create *path* recursively if it does not already exist."""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# wafer_defect_detection package root (one level up from io_utils/)
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
# All artifacts live under wafer_defect_detection/result/ unless WDD_OUTDIR overrides it.
_OUT_ROOT = os.getenv("WDD_OUTDIR", os.path.join(PROJECT_ROOT, "result"))

# Canonical sub‑folders
_DEBUG_DIR = os.path.join(_OUT_ROOT, "debug_img")
_CSV_DIR = os.path.join(_OUT_ROOT, "csv")

# ---------------------------------------------------------------------
# Per‑run timestamped directory (created on first import)
# ---------------------------------------------------------------------
_RUN_DIR = os.path.join(
    _OUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S")
)
ensure_dir(_RUN_DIR)

# Sub‑folder for debug images inside the run directory
_DEBUG_RUN_DIR = os.path.join(_RUN_DIR, "debug_img")
ensure_dir(_DEBUG_RUN_DIR)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _timestamp() -> str:
    """ISO‑like timestamp safe for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def debug_img_path(filename: str | None = None) -> str:
    """
    Path under the **current run** debug_img folder.
    If *filename* is None, a timestamped PNG name is generated.
    """
    if filename is None:
        filename = f"{_timestamp()}.png"
    path = os.path.join(_DEBUG_RUN_DIR, filename)
    ensure_dir(os.path.dirname(path))
    return path

def csv_path(filename: str) -> str:
    """Absolute path inside ``results/csv`` subdir."""
    ensure_dir(_CSV_DIR)
    return os.path.join(_CSV_DIR, filename)

def custom_out_path(*relative_parts: str) -> str:
    """
    Join *relative_parts* under the configured results root, ensuring
    parent directories exist.
    """
    path = os.path.join(_OUT_ROOT, *relative_parts)
    ensure_dir(os.path.dirname(path))
    return path

# ---------------------------------------------------------------------
# Run‑scoped helpers
# ---------------------------------------------------------------------
def run_path(*relative_parts: str) -> str:
    """
    Join *relative_parts* under the timestamped run directory.
    Parent folders are auto‑created.
    """
    path = os.path.join(_RUN_DIR, *relative_parts)
    ensure_dir(os.path.dirname(path))
    return path


# ---------------------------------------------------------------------
# Additional run-scoped helpers
# ---------------------------------------------------------------------
def run_debug_path(filename: str | None = None) -> str:
    """
    Convenience wrapper generating a path under the per-run ``debug_img`` folder.

    If *filename* is omitted a timestamped PNG name is produced.  Ensures that
    the parent directory exists, and returns the absolute path.
    """
    if filename is None:
        filename = f"{_timestamp()}.png"
    return run_path("debug_img", filename)


def run_log_path(name: str = "run.log") -> str:
    """Canonical location for the per‑run log file."""
    return run_path(name)

__all__ = [
    "PROJECT_ROOT",
    "debug_img_path",
    "csv_path",
    "custom_out_path",
    "run_path",
    "run_log_path",
    "run_debug_path",
    "ensure_dir",
]
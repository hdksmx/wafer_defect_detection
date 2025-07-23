"""timing.py — lightweight stopwatch utilities for step‑wise profiling

Designed for wafer‑scratch detection pipeline but generic enough for any CLI script.

Features
--------
* **Timer** context‑manager – `with Timer("stage name"):`
* **timeit decorator** – `@timeit()` for functions
* Global accumulation per label & pretty summary printer
* Zero external dependencies (standard library only)

Example
-------
```python
from timing import Timer, timeit

@timeit()            # times the whole function
def heavy_func():
    ...

with Timer("preprocess"):
    preprocess(img)

Timer.summary()       # prints cumulative timings
```
"""

from __future__ import annotations


import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Dict, Optional

__all__ = ["Timer", "timeit"]

# ----------------------------------------------------------------------------
# Core timer
# ----------------------------------------------------------------------------

class Timer:
    """Context‑manager style stopwatch with global accumulation.

    Parameters
    ----------
    label : str
        Human‑readable name for the timed section.
    accumulate : bool, default True
        If *True*, elapsed seconds are added to an internal accumulator
        keyed by *label* so you can print a summary later.
    silent : bool, default False
        If *True*, do **not** print end‑of‑block timing message (useful inside
        decorator).
    """

    _acc: Dict[str, float] = defaultdict(float)  # cumulative seconds per label

    def __init__(self, label: str, *, accumulate: bool = True, silent: bool = False):
        self.label = label
        self.accumulate = accumulate
        self.silent = silent
        self._start: float | None = None
        self.elapsed: float | None = None

    # ---------------------------------------------------------------------
    # Context‑manager protocol
    # ---------------------------------------------------------------------

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D401
        if self._start is None:
            raise RuntimeError("Timer was never started — did you call __enter__()?")
        self.elapsed = time.perf_counter() - self._start
        if self.accumulate:
            Timer._acc[self.label] += self.elapsed
        if not self.silent:
            self._print_single(self.label, self.elapsed)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------

    # seconds → ms string
    @staticmethod
    def _fmt(seconds: float) -> str:
        return f"{seconds*1000:.2f} ms"

    @staticmethod
    def _print_single(label: str, seconds: float) -> None:
        print(f"[TIMER] {label:<20s}: {Timer._fmt(seconds)}")

    # Public API ---------------------------------------------------------

    @staticmethod
    def stats() -> Dict[str, float]:
        """Return *copy* of cumulative timing dict {label: seconds}."""
        return dict(Timer._acc)

    @staticmethod
    def summary(reset: bool = False) -> None:
        """Pretty‑print cumulative statistics and optionally reset them."""
        if not Timer._acc:
            print("[TIMER] No measurements recorded.")
            return

        width = max(len(k) for k in Timer._acc)
        print("\n========== Timing Summary ==========")
        total = 0.0
        for lbl, sec in Timer._acc.items():
            total += sec
            print(f"{lbl.ljust(width)} : {Timer._fmt(sec)}")
        print("-" * (width + 30))
        print(f"{'Total'.ljust(width)} : {Timer._fmt(total)}")
        print("====================================\n")

        if reset:
            Timer.reset()

    @staticmethod
    def reset() -> None:
        """Clear all accumulated statistics."""
        Timer._acc.clear()


# ----------------------------------------------------------------------------
# Decorator convenience
# ----------------------------------------------------------------------------

def timeit(label: Optional[str] = None, *, accumulate: bool = True) -> Callable[[Callable[..., "T"]], Callable[..., "T"]]:
    """Decorator to time a function call.

    Parameters
    ----------
    label : str | None
        Label under which to accumulate time. If *None*, `func.__name__` is used.
    accumulate : bool
        Accumulate time in :pyclass:`Timer` global stats.
    """

    def decorator(func: Callable[..., "T"]) -> Callable[..., "T"]:
        name = label or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(name, accumulate=accumulate, silent=True):
                return func(*args, **kwargs)

        return wrapper

    return decorator



#!/usr/bin/env python3
"""
cli.py
======

Thin command‑line wrapper for the wafer‑scratch detection pipeline.

This script does **no** argument parsing of its own; it simply delegates
to :pyfunc:`wafer_defect_detection.pipeline.main`, so all options and
documentation are maintained in one place.

Examples
--------
Detect scratches and write results:

    python -m wafer_defect_detection.cli \
        --img sample/wafer.png \
        --out results/mask.png \
        --overlay results/overlay.png \
        --preprocess gauss_clahe \
        --debug

The full list of arguments can be viewed with ``-h`` or ``--help``.
"""
from __future__ import annotations

import sys

import pipeline


def main() -> None:
    """Forward ``sys.argv[1:]`` to :pyfunc:`pipeline.main`."""
    pipeline.main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
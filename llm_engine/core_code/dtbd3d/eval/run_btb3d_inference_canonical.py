#!/usr/bin/env python3
"""Deprecated wrapper for run_report_generation.py."""

from __future__ import annotations

from pathlib import Path
import sys

CORE_CODE = Path(__file__).resolve().parents[2]
if str(CORE_CODE) not in sys.path:
    sys.path.insert(0, str(CORE_CODE))

from dtbd3d.eval.run_report_generation import main


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Recompute the benchmark's simulated test-set numbers from the shipped predictions (thin wrapper).

Runs scripts/recompute_benchmark.py of this repository with the same arguments. It reads results/test_v1/ and
results/truth/, recomputes every metric with src/eval/metrics.py, compares each value with the test run's own
benchmark.json, descriptive.json and summary.md (tolerance 1e-6), and writes results/recompute/ by default.
Exit status 0 iff every compared value agrees.

  python mvp/recompute_benchmark.py                                   # everything (about 1-2 min)
  python mvp/recompute_benchmark.py --methods coloc_abf,coloc_susie_mm --modes least_favourable --out /tmp/x
  python mvp/recompute_benchmark.py --help                            # all options
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recompute_benchmark.py"

if __name__ == "__main__":
    sys.argv[0] = str(SCRIPT)
    runpy.run_path(str(SCRIPT), run_name="__main__")

"""Shared fixtures for baseline-wrapper tests: toy loci from a real 1000G chr22 NFE LD window."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

assert os.environ.get("COLOC_ENV_ACTIVE") == "1", "source env/activate.sh first"

from baselines.toy import KG, load_window, make_toy_locus, pick_pair  # noqa: E402
from common.seeds import VALID_SEED, rng  # noqa: E402

HAVE_KG = (KG / "chr22.pgen").exists()
needs_kg = pytest.mark.skipif(not HAVE_KG, reason="data/processed/1kg/chr22.pgen missing")


@pytest.fixture(scope="session")
def window300():
    if not HAVE_KG:
        pytest.skip("chr22 pgen missing")
    return load_window(300)


@pytest.fixture(scope="session")
def controls(window300):
    """Known-direction controls (lambda = 30): shared, distinct (r^2 < 0.1), null trait 2; two replicates each."""
    X, maf, ids, pos = window300
    R = X.T @ X / X.shape[0]
    out = {}
    for rep in range(2):
        g = rng(VALID_SEED, "tests/baselines/controls", rep)
        a, b = pick_pair(R, g, r2_max=0.1)
        out[("shared", rep)] = make_toy_locus(X, maf, ids, pos, {a: 30.0}, {a: 30.0}, g, locus_id=f"shared{rep}")
        out[("distinct", rep)] = make_toy_locus(X, maf, ids, pos, {a: 30.0}, {b: 30.0}, g, locus_id=f"distinct{rep}")
        out[("null2", rep)] = make_toy_locus(X, maf, ids, pos, {a: 30.0}, {}, g, locus_id=f"null2{rep}")
        assert R[a, b] ** 2 < 0.1
    return out

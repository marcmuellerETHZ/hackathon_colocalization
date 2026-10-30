"""Truth parquet reader/writer (INTERFACES §2).

!!! Method code (src/fam*/, src/baselines/) must NEVER import this module or open *.truth.parquet. !!!
A static test (tests/eval/test_truth_isolation.py) scans those directories, and `read_truth` refuses to run
when called from a module in those packages.

One row per locus. Columns (see TRUTH_SCHEMA): the oracle class and the quantities metrics need.
List-valued columns (causal indices/effects) are optional and stored as parquet list columns.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from eval.io import guard_locus

ORACLE_CLASSES = ("S", "D", "W", "G", "U")

# name -> (required?, kind, meaning)
TRUTH_SCHEMA = {
    "locus_id": (True, "str", "locus group name in the HDF5 locus set"),
    "oracle_class": (True, "str", "S/D/W/G/U (DESIGN §1, precedence-ordered partition)"),
    "mu1": (True, "float", "class-determining realised noncentrality, trait 1 (DESIGN §1; 0 if none)"),
    "mu2": (True, "float", "class-determining realised noncentrality, trait 2"),
    "r2": (True, "float", "r² (R_true) of the class-determining D/U pair; NaN otherwise"),
    "d": (True, "float", "identifiability index of the class-determining D/U pair; NaN otherwise"),
    "rho": (True, "float", "true realised null correlation rho"),
    "block_id": (True, "str", "LD block containing the window anchor (TSS/lead): the bootstrap unit"),
    "cell_id": (True, "str", "evaluation cell id (datasets/configs/testset_v1.yaml naming)"),
    "split": (True, "str", "train/validation/test"),
    "mismatch_level": (False, "str", "L0..L6"),
    "pairing_type": (False, "str", "P1/P2/P3"),
    "overlap": (False, "float", "grid overlap parameter of the cell (r_p for P1, rho for P2/P3)"),
    "control": (False, "str", "'' or a control type: 'pos_shared' (lambda=30 shared, r²=1) / 'null'"),
    "causal1": (False, "list", "causal variant indices, trait 1"),
    "causal2": (False, "list", "causal variant indices, trait 2"),
    "effect1": (False, "list", "own-effect lambda per causal variant, trait 1"),
    "effect2": (False, "list", "own-effect lambda per causal variant, trait 2"),
    "mu1_all": (False, "list", "realised mu per causal variant, trait 1"),
    "mu2_all": (False, "list", "realised mu per causal variant, trait 2"),
}
REQUIRED = [k for k, v in TRUTH_SCHEMA.items() if v[0]]

_FORBIDDEN_CALLER_PREFIXES = ("fam", "baselines")


def _check_caller() -> None:
    for fr in inspect.stack()[2:]:
        mod = fr.frame.f_globals.get("__name__", "") or ""
        top = mod.split(".")[0]
        if any(top.startswith(p) for p in _FORBIDDEN_CALLER_PREFIXES):
            raise PermissionError(f"truth files must not be read by method code (caller module {mod!r})")


def validate_truth(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"truth table missing required columns {missing}")
    if df["locus_id"].duplicated().any():
        raise ValueError("duplicate locus_id in truth table")
    bad = set(df["oracle_class"].unique()) - set(ORACLE_CLASSES)
    if bad:
        raise ValueError(f"unknown oracle classes {bad}")
    if df["block_id"].isna().any() or df["cell_id"].isna().any():
        raise ValueError("block_id and cell_id must be non-null")


def _guard(df: pd.DataFrame) -> None:
    for lid, sp, b in zip(df["locus_id"], df["split"], df["block_id"]):
        guard_locus(str(lid), str(sp), (str(b),))


def write_truth(df: pd.DataFrame, path: str | Path, guard: bool = True) -> None:
    validate_truth(df)
    if guard:
        _guard(df)
    out = df.copy()
    for c in ("locus_id", "oracle_class", "block_id", "cell_id", "split"):
        out[c] = out[c].astype(str)
    out.to_parquet(path, index=False)


def read_truth(path: str | Path, guard: bool = True) -> pd.DataFrame:
    _check_caller()
    df = pd.read_parquet(path)
    validate_truth(df)
    if guard:
        _guard(df)
    for c in ("mu1", "mu2", "r2", "d", "rho"):
        df[c] = df[c].astype(np.float64)
    return df

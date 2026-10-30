"""LocusData access for the baseline wrappers (INTERFACES §2 / §4).

The wrappers duck-type the locus object: anything with the INTERFACES §2 fields works. The canonical class is
`eval.io.LocusData` (owner: eval agent). It is imported lazily; if it is not importable, the minimal
STUB below (clearly marked, same field names) is used so that this module and its tests can run on their own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------------------------------------------
# STUB (baseline agent): used only when eval.io.LocusData is not importable. Field names mirror INTERFACES §2.
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class _StubLocusData:
    locus_id: str
    variant_id: np.ndarray
    pos: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    beta1: np.ndarray
    se1: np.ndarray
    beta2: np.ndarray
    se2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    maf1: np.ndarray
    maf2: np.ndarray
    maf_ref: np.ndarray
    ld_type: str
    n_ref: int
    R: np.ndarray | None = None
    X_ref: np.ndarray | None = None
    rho_hat: float = float("nan")
    rho_se: float = float("nan")
    split: str = ""
    block_ids: tuple = ()
    mismatch_level: str = ""
    pairing_type: str = ""
    cell_id: str = ""
    window_chr: str = ""
    window_start: int = -1
    window_end: int = -1
    extra_attrs: dict = field(default_factory=dict)


def locus_class():
    """eval.io.LocusData if importable, else the stub."""
    try:
        from eval.io import LocusData  # noqa: WPS433 (lazy import by design)
        return LocusData
    except Exception:  # pragma: no cover - depends on the eval module being present
        return _StubLocusData


def make_locus(**kw):
    """Construct a LocusData (canonical class if available)."""
    cls = locus_class()
    names = set(getattr(cls, "__dataclass_fields__", {}).keys())
    extra = {k: kw.pop(k) for k in list(kw) if names and k not in names}
    obj = cls(**kw)
    if extra:
        getattr(obj, "extra_attrs", {}).update(extra)
    return obj


# ---------------------------------------------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------------------------------------------
def p_of(locus) -> int:
    return int(len(locus.z1))


def dense_ld(locus) -> np.ndarray:
    """Dense inference LD as float64 (p x p), symmetrised. For factor form R = X_ref' X_ref / n_ref.

    The result is cached on the locus object (attribute `_baselines_R64`) so that several wrappers applied to
    the same object build it once; wrappers count the build in their own runtime only on first use (the
    harness gives each method its own fresh locus object, so per-method runtimes include LD loading).
    """
    cached = getattr(locus, "_baselines_R64", None)
    if cached is not None:
        return cached
    if locus.ld_type == "dense":
        R = np.asarray(locus.R, dtype=np.float64)
    elif locus.ld_type == "factor":
        X = np.asarray(locus.X_ref, dtype=np.float64)
        R = X.T @ X / float(locus.n_ref)
    else:
        raise ValueError(f"unknown ld_type {locus.ld_type!r}")
    R = 0.5 * (R + R.T)
    try:
        object.__setattr__(locus, "_baselines_R64", R)
    except Exception:  # pragma: no cover
        pass
    return R


def scalar_n(arr) -> float:
    """Per-trait sample size given to methods that take one N: the median of the per-SNP N (constant unless L5)."""
    return float(np.median(np.asarray(arr, dtype=np.float64)))


def rho_hat(locus) -> float:
    """rho_hat attribute; NaN / missing -> 0.0 (documented in README: methods then assume independence)."""
    v = getattr(locus, "rho_hat", float("nan"))
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not math.isfinite(v) else v


def n_ref_of(locus) -> int:
    return int(locus.n_ref)


def snp_names(locus) -> list[str]:
    """Unique SNP names (variant ids; index-suffixed if duplicated, which QC should prevent)."""
    v = [str(x) for x in locus.variant_id]
    if len(set(v)) != len(v):
        v = [f"{x}#{i}" for i, x in enumerate(v)]
    return v

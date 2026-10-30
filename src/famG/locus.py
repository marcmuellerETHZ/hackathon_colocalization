"""STUB (owner: famA/famG agent) -- minimal stand-in for src/eval/io.py's LocusData (INTERFACES §2, §4).

src/eval/io.py (eval agent) did not exist when Families A/G were built. This dataclass mirrors the
locus-set fields of INTERFACES §2 so that the Methods can be exercised and tested now. When the eval
agent's LocusData lands, the Methods accept it unchanged as long as it exposes the same attribute
names (duck typing); this stub should then be deleted. Nothing here reads truth files.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LocusData:
    locus_id: str
    z1: np.ndarray
    z2: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    ld_type: str = "dense"               # "dense" | "factor"
    R: np.ndarray | None = None          # (p,p) float32
    X_ref: np.ndarray | None = None      # (n_ref,p) float32 standardized dosages; R = X^T X / n_ref
    n_ref: int | None = None
    beta1: np.ndarray | None = None
    se1: np.ndarray | None = None
    beta2: np.ndarray | None = None
    se2: np.ndarray | None = None
    maf1: np.ndarray | None = None
    maf2: np.ndarray | None = None
    maf_ref: np.ndarray | None = None
    variant_id: np.ndarray | None = None
    pos: np.ndarray | None = None
    rho_hat: float = 0.0
    rho_se: float = float("nan")
    attrs: dict = field(default_factory=dict)

    @property
    def p(self) -> int:
        return len(self.z1)

    def ld_input(self):
        """('dense', R) or ('factor', X_ref) as consumed by famG.susie_rss.susie_rss_batch."""
        if self.ld_type == "factor":
            return "factor", self.X_ref
        return "dense", self.R


def load_locus_set(path: str, locus_ids: list[str] | None = None) -> list[LocusData]:
    """Read an INTERFACES §2 HDF5 locus set (one group per locus)."""
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        ids = locus_ids if locus_ids is not None else list(f.keys())
        for lid in ids:
            g = f[lid]
            get = lambda k: np.asarray(g[k]) if k in g else None
            ld_type = g.attrs.get("ld_type", "dense")
            if isinstance(ld_type, bytes):
                ld_type = ld_type.decode()
            out.append(LocusData(
                locus_id=lid, z1=get("z1"), z2=get("z2"), n1=get("n1"), n2=get("n2"), ld_type=ld_type,
                R=get("R"), X_ref=get("X_ref"), n_ref=int(g.attrs["n_ref"]) if "n_ref" in g.attrs else None,
                beta1=get("beta1"), se1=get("se1"), beta2=get("beta2"), se2=get("se2"),
                maf1=get("maf1"), maf2=get("maf2"), maf_ref=get("maf_ref"), variant_id=get("variant_id"), pos=get("pos"),
                rho_hat=float(g.attrs.get("rho_hat", 0.0)), rho_se=float(g.attrs.get("rho_se", np.nan)),
                attrs={k: g.attrs[k] for k in g.attrs.keys()},
            ))
    return out

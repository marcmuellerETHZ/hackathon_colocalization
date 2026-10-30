"""Strength-only control scorer (decision D6(e)); hash-protected via configs/FROZEN_SHA256.

Why: 5x5 power matching removes most but not all power confounding (a power-only scorer kept AUROC 0.56 after
matching in the M3 toy check). Every benchmark therefore also scores a reference "method" that sees ONLY the
observed signal strength of each trait, and claims are read against it.

Features (observed data only, never truth): lead |z| of trait 1 and of trait 2 (max_j |z_tj|), on the log scale.
Table: per stratum (default: the locus `pairing_type`, plus a pooled "__all__" stratum used for unseen strata),
n_bins x n_bins quantile bins of the two features (edges from the fitting loci), and class counts per bin.
Scores, with pseudo-count alpha shrinking each bin towards the stratum's class shares pi:
    P(c | bin) = (n_c,bin + alpha * pi_c) / (n_bin + alpha),   c in {S, D, W, other (G/U)}
    P_S = P(S | bin),  P_D = P(D | bin),  P_W = P(W | bin),  s = P_S / (P_S + P_D) = P(S | bin, S ∪ D).
s is a monotone function of the strength-only likelihood ratio of S vs D within the bin, so its ranking does not
depend on class proportions of the fitting set.

Fitting needs truth labels and is done by evaluation code only (scripts/run_benchmark.py):
  * validation benchmark: K-fold CROSS-FITTING on the benchmark set itself (fold = sha256(locus_id) mod K; loci are
    independent simulation draws, so locus-level folds do not leak labels);
  * later runs (e.g. the test run): a single table fitted on a validation set, passed as a JSON file.
The scorer itself (reference_methods.StrengthOnly) reads only the table and the locus z-scores.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eval.adapters import ratio_or_half

CLASSES = ("S", "D", "W", "other")
POOLED = "__all__"


def lead_features(z1, z2) -> tuple[float, float]:
    return float(np.max(np.abs(np.asarray(z1, float)))), float(np.max(np.abs(np.asarray(z2, float))))


def read_features(locus_set: str | Path, ids=None, by: str = "pairing_type") -> pd.DataFrame:
    """Fast feature read (z1, z2 and attributes only; no LD). Applies the test-split guard via read_locus_attrs."""
    from eval.io import read_locus_attrs
    attrs = read_locus_attrs(locus_set)  # guard on every locus
    keep = attrs["locus_id"].astype(str).tolist() if ids is None else [str(i) for i in ids]
    strata = dict(zip(attrs["locus_id"].astype(str), attrs[by].astype(str) if by in attrs else [""] * len(attrs)))
    rows = []
    with h5py.File(locus_set, "r") as f:
        for lid in keep:
            g = f[lid]
            a1, a2 = lead_features(g["z1"][()], g["z2"][()])
            rows.append((lid, a1, a2, strata.get(lid, "")))
    return pd.DataFrame(rows, columns=["locus_id", "lead_abs_z1", "lead_abs_z2", "stratum"])


def fold_of(locus_id: str, k: int) -> int:
    return int.from_bytes(hashlib.sha256(str(locus_id).encode()).digest()[:8], "little") % int(k)


def _cls4(c) -> np.ndarray:
    c = np.asarray(c).astype(str)
    return np.where(np.isin(c, ("S", "D", "W")), c, "other")


@dataclass
class StrengthTable:
    n_bins: int = 8
    alpha: float = 5.0
    edges: dict = field(default_factory=dict)   # stratum -> [edges1 (n_bins-1), edges2 (n_bins-1)] on log|z|
    counts: dict = field(default_factory=dict)  # stratum -> nested list [n_bins][n_bins][4]
    prior: dict = field(default_factory=dict)   # stratum -> [4] class shares
    n_fit: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit / predict
    @classmethod
    def fit(cls, feats: pd.DataFrame, classes, n_bins: int = 8, alpha: float = 5.0) -> "StrengthTable":
        t = cls(n_bins=n_bins, alpha=alpha)
        c4 = _cls4(classes)
        x1 = np.log(np.maximum(feats["lead_abs_z1"].to_numpy(float), 1e-12))
        x2 = np.log(np.maximum(feats["lead_abs_z2"].to_numpy(float), 1e-12))
        st = feats["stratum"].astype(str).to_numpy()
        groups = {POOLED: np.ones(len(feats), bool)}
        for s in sorted(set(st)):
            groups[s] = st == s
        q = np.linspace(0, 1, n_bins + 1)[1:-1]
        for s, m in groups.items():
            if not m.any():
                continue
            e1, e2 = np.quantile(x1[m], q), np.quantile(x2[m], q)
            k1, k2 = np.searchsorted(e1, x1[m], side="right"), np.searchsorted(e2, x2[m], side="right")
            cnt = np.zeros((n_bins, n_bins, 4))
            ci = np.array([CLASSES.index(c) for c in c4[m]], dtype=int)
            np.add.at(cnt, (k1, k2, ci), 1.0)
            t.edges[s] = [e1.tolist(), e2.tolist()]
            t.counts[s] = cnt.tolist()
            t.prior[s] = (cnt.sum(axis=(0, 1)) / cnt.sum()).tolist()
            t.n_fit[s] = int(m.sum())
        return t

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        x1 = np.log(np.maximum(feats["lead_abs_z1"].to_numpy(float), 1e-12))
        x2 = np.log(np.maximum(feats["lead_abs_z2"].to_numpy(float), 1e-12))
        st = feats["stratum"].astype(str).to_numpy()
        P = np.full((len(feats), 4), np.nan)
        used = np.empty(len(feats), dtype=object)
        for s in sorted(set(st)):
            m = st == s
            key = s if s in self.counts else POOLED
            e1, e2 = (np.asarray(e) for e in self.edges[key])
            cnt, pi = np.asarray(self.counts[key]), np.asarray(self.prior[key])
            k1, k2 = np.searchsorted(e1, x1[m], side="right"), np.searchsorted(e2, x2[m], side="right")
            nb = cnt[k1, k2]  # (n, 4)
            P[m] = (nb + self.alpha * pi) / (nb.sum(axis=1, keepdims=True) + self.alpha)
            used[m] = key
        out = pd.DataFrame({"locus_id": feats["locus_id"].astype(str).to_numpy(), "P_S": P[:, 0], "P_D": P[:, 1],
                            "P_W": P[:, 2], "table_stratum": used})
        out["s"] = [ratio_or_half(a, b) for a, b in zip(P[:, 0], P[:, 1])]
        return out

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"kind": "single", "n_bins": self.n_bins, "alpha": self.alpha, "edges": self.edges,
                "counts": self.counts, "prior": self.prior, "n_fit": self.n_fit}

    @classmethod
    def from_dict(cls, d: dict) -> "StrengthTable":
        return cls(n_bins=int(d["n_bins"]), alpha=float(d["alpha"]), edges=d["edges"], counts=d["counts"],
                   prior=d["prior"], n_fit=d.get("n_fit", {}))


def fit_crossfit(feats: pd.DataFrame, classes, k: int = 5, n_bins: int = 8, alpha: float = 5.0) -> dict:
    """K tables; table j is fitted on loci with fold != j and is used to score fold j."""
    folds = np.array([fold_of(l, k) for l in feats["locus_id"]])
    classes = np.asarray(classes)
    tabs = [StrengthTable.fit(feats[folds != j], classes[folds != j], n_bins, alpha).to_dict() for j in range(k)]
    return {"kind": "crossfit", "k": int(k), "tables": tabs}


def load(spec: str | Path | dict) -> dict:
    return spec if isinstance(spec, dict) else json.loads(Path(spec).read_text())


def predict_any(table: dict, feats: pd.DataFrame) -> pd.DataFrame:
    """Score features with a single table or a cross-fitted set (each locus by the table of its own fold)."""
    if table["kind"] == "single":
        out = StrengthTable.from_dict(table).predict(feats)
        out["fold"] = -1
        return out
    if table["kind"] != "crossfit":
        raise ValueError(table["kind"])
    k = int(table["k"])
    folds = np.array([fold_of(l, k) for l in feats["locus_id"]])
    parts = []
    for j in range(k):
        m = folds == j
        if m.any():
            p = StrengthTable.from_dict(table["tables"][j]).predict(feats[m])
            p["fold"] = j
            parts.append(p)
    out = pd.concat(parts).set_index("locus_id").loc[feats["locus_id"].astype(str)].reset_index()
    return out


def dump(table: dict, path: str | Path) -> str:
    """Write the table JSON; returns its sha256 (recorded in run manifests)."""
    txt = json.dumps(table, sort_keys=True)
    Path(path).write_text(txt)
    return hashlib.sha256(txt.encode()).hexdigest()

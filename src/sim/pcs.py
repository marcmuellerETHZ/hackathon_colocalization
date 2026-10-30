"""Genome-wide PC panel for the L3 admixed study (DESIGN v4 §2.3: "z is computed with the top 2 genotype PCs of
the study sample as covariates").

Panel: N_PC_SNPS autosomal SNPs with MAF >= 0.05 in pool A and in ASW/ACB, drawn at random (seed 20261026) and
thinned to >= 500 kb apart (approximately LD-independent). The NFE and ADM mosaic populations are materialised
at these SNPs (the mosaic is window-independent, so these are the same synthetic individuals as in every
window). Cached in data/processed/sim/pc_panel_<version>.npz (uint8 dosages), keyed by world version and the
populations' N_e.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pgenlib
import torch

from common.seeds import SPLIT_SEED, rng
from data.variants import OUT_DIR as VAR_DIR, PGEN_DIR, read_psam
from sim import params
from sim.mosaic import Mosaic
from sim.world import interp_cm, pools

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/sim"
MIN_SPACING = 500_000


def panel_variants(n: int = params.N_PC_SNPS) -> pd.DataFrame:
    g = rng(SPLIT_SEED, "pc_panel", params.WORLD_VERSION)
    rows = []
    for c in range(1, 23):
        v = pd.read_parquet(VAR_DIR / f"chr{c}.parquet", columns=["chr", "pos", "variant_id", "pgen_idx", "af_A", "af_ASW_ACB"])
        mA = np.minimum(v["af_A"], 1 - v["af_A"])
        mB = np.minimum(v["af_ASW_ACB"], 1 - v["af_ASW_ACB"])
        v = v[(mA >= 0.05) & (mB >= 0.05)]
        rows.append(v)
    v = pd.concat(rows, ignore_index=True)
    v = v.iloc[np.sort(g.choice(len(v), min(len(v), 20 * n), replace=False))]
    keep, last = [], {}
    for i, (c, p) in enumerate(zip(v["chr"], v["pos"])):
        if p - last.get(c, -10 ** 12) >= MIN_SPACING:
            keep.append(i)
            last[c] = p
    v = v.iloc[keep]
    v = v.iloc[np.sort(g.choice(len(v), min(n, len(v)), replace=False))].reset_index(drop=True)
    return v


def _pool_haplotypes(chrom: str, pgen_idx: np.ndarray, samples: list[str]) -> np.ndarray:
    psam = read_psam(PGEN_DIR / f"{chrom}.psam")
    pos = {s: i for i, s in enumerate(psam)}
    idx = np.array([pos[s] for s in samples], dtype=np.uint32)
    order = np.argsort(idx, kind="stable")
    subset = idx[order]
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    r = pgenlib.PgenReader(bytes(str(PGEN_DIR / f"{chrom}.pgen"), "utf8"), raw_sample_ct=len(psam), sample_subset=subset)
    n = len(subset)
    al = np.empty((len(pgen_idx), 2 * n), dtype=np.int32)
    pp = np.empty((len(pgen_idx), n), dtype=np.uint8)
    r.read_alleles_and_phasepresent_list(np.asarray(pgen_idx, dtype=np.uint32), al, pp)
    r.close()
    assert (al >= 0).all()
    het = al[:, 0::2] != al[:, 1::2]
    assert pp[het].all(), "unphased heterozygous calls"
    al = al.reshape(len(pgen_idx), n, 2)[:, back, :].reshape(len(pgen_idx), 2 * n)
    return np.ascontiguousarray(al.T.astype(np.int8))


def build_pc_panel(device="cuda") -> dict[str, np.ndarray]:
    v = panel_variants()
    out = {}
    for name in ("NFE", "ADM"):
        m = Mosaic.named(name)
        parts = []
        for c, vc in v.groupby("chr", sort=False):
            vc = vc.sort_values("pos").reset_index(drop=True)
            vc["cm"] = interp_cm(c, vc["pos"].to_numpy())
            Hs = _pool_haplotypes(c, vc["pgen_idx"].to_numpy(), pools()[m.pool])
            h = m.haplotypes_from_source(c, Hs, vc, device)
            parts.append((h[0::2] + h[1::2]).cpu().numpy().astype(np.uint8))
        out[name] = np.concatenate(parts, axis=1)
    out["variant_id"] = v["variant_id"].to_numpy().astype(str)   # v is in (chr1..22, pos) order = groupby order
    return out


def _cache_path() -> Path:
    ne = (params.POP_NE.get("NFE", params.NE), params.POP_NE.get("ADM", params.NE))
    return CACHE / f"pc_panel_{params.WORLD_VERSION}_ne{int(ne[0])}_{int(ne[1])}.npz"


def load_pc_panel(device="cuda") -> dict[str, torch.Tensor]:
    f = _cache_path()
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        d = build_pc_panel(device)
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, **d)
        tmp.rename(f)
    d = np.load(f)
    return {k: torch.as_tensor(d[k].astype(np.float32), device=device) for k in ("NFE", "ADM")}

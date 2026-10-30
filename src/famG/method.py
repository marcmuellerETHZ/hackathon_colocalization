"""Family G Method (INTERFACES §4): GPU coloc-SuSiE = batched SuSiE-RSS (susieR 0.14.2 semantics) per trait
+ coloc.bf_bf over credible-set effects, with the DESIGN §4.3 coloc-SuSiE adapter and coloc.abf fallback."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

from .coloc import bf_bf_batch, coloc_abf
from .susie_rss import SusieRSSConfig, susie_rss_batch


def ld_input(lc):
    """('dense', R) or ('factor', X_ref) from any LocusData-like object (src/eval/io.LocusData or famG.locus stub)."""
    if getattr(lc, "ld_type", "dense") == "factor":
        return "factor", lc.X_ref
    return "dense", lc.R


def _scalar_n(n) -> float:
    n = np.atleast_1d(np.asarray(n, dtype=np.float64))
    return float(np.median(n))


def _beta_se(z, n, maf, beta, se):
    if beta is not None and se is not None:
        return np.asarray(beta, float), np.asarray(se, float)
    if maf is None:
        return None, None
    f = np.asarray(maf, float)
    s = 1.0 / np.sqrt(2.0 * np.asarray(n, float) * f * (1 - f))   # DESIGN §2.2, sdY = 1
    return np.asarray(z, float) * s, s


def coloc_susie_adapter(pp_pairs: np.ndarray | None, pp_abf: np.ndarray | None):
    """DESIGN §4.3 coloc-SuSiE row via the single shared implementation eval.adapters.coloc_susie:
    s = max PP4/(PP3+PP4); P_S = max PP4; P_D = (1 - max PP4) * max PP3; P_W = 1 - P_S - P_D;
    coloc.abf row (PP4/(PP3+PP4), PP4, PP3, PP0+PP1+PP2) when a trait has no CS."""
    from eval import adapters
    fallback = pp_pairs is None or np.asarray(pp_pairs).size == 0
    sc = adapters.coloc_susie(None if fallback else np.asarray(pp_pairs).reshape(-1, 5),
                              None if pp_abf is None else np.asarray(pp_abf).reshape(5))
    return sc["s"], sc["P_S"], sc["P_D"], sc["P_W"], fallback


class GPUColocSusie:
    name = "famG_coloc_susie_gpu"

    def __init__(self, cfg: SusieRSSConfig | None = None, p1=1e-4, p2=1e-4, p12=5e-6, p12_abf=None, W_abf=0.15,
                 device=None):
        """p12 is used by coloc.bf_bf and, unless p12_abf is given, by the coloc.abf fallback (the tier-1
        coloc-SuSiE wrapper src/baselines/methods.py:ColocSusie uses its one p12 for both and W = 0.15)."""
        self.cfg = cfg or SusieRSSConfig()
        self.p1, self.p2, self.p12 = p1, p2, p12
        self.p12_abf = p12 if p12_abf is None else p12_abf
        self.W_abf = W_abf
        self.device = device

    def fit_batch(self, loci: list) -> list[dict]:
        t0 = time.time()
        B = len(loci)
        zs, lds, ns, kinds, nrefs = [], [], [], [], []
        for t in (1, 2):
            for lc in loci:
                kind, ld = ld_input(lc)
                zs.append(np.asarray(getattr(lc, f"z{t}"), dtype=np.float64))
                lds.append(ld); kinds.append(kind)
                ns.append(_scalar_n(getattr(lc, f"n{t}")))
                nrefs.append(lc.n_ref if kind == "factor" else None)
        fits = susie_rss_batch(zs, lds, ns, kinds, self.cfg, device=self.device, n_refs=nrefs)
        f1, f2 = fits[:B], fits[B:]
        bf1 = [f["lbf_variable"][f["sets"]["cs_index"]] if f["sets"]["cs_index"] else np.zeros((0, len(f["pip"]))) for f in f1]
        bf2 = [f["lbf_variable"][f["sets"]["cs_index"]] if f["sets"]["cs_index"] else np.zeros((0, len(f["pip"]))) for f in f2]
        pps = bf_bf_batch(bf1, bf2, self.p1, self.p2, self.p12, device=self.device)
        rt = (time.time() - t0) / max(B, 1)
        out = []
        for i, lc in enumerate(loci):
            a, b = f1[i], f2[i]
            status = "ok"
            if a["status"] != "ok" or b["status"] != "ok":
                status = "failed"
            elif not (a["converged"] and b["converged"]):
                status = "nonconverged"
            pp_abf = None
            if pps[i].size == 0:
                b1, s1 = _beta_se(lc.z1, lc.n1, getattr(lc, "maf1", None), getattr(lc, "beta1", None), getattr(lc, "se1", None))
                b2, s2 = _beta_se(lc.z2, lc.n2, getattr(lc, "maf2", None), getattr(lc, "beta2", None), getattr(lc, "se2", None))
                if b1 is None or b2 is None:
                    out.append(dict(locus_id=lc.locus_id, method=self.name, s=math.nan, P_S=math.nan, P_D=math.nan,
                                    P_W=math.nan, E1=math.nan, E2=math.nan, status="failed", runtime_s=rt,
                                    extra=json.dumps({"reason": "no CS and no beta/se/maf for coloc.abf fallback"})))
                    continue
                pp_abf = coloc_abf(b1, s1, b2, s2, self.p1, self.p2, self.p12_abf, W=self.W_abf)
            s, P_S, P_D, P_W, fb = coloc_susie_adapter(pps[i] if pps[i].size else None, pp_abf)
            if status == "failed":
                s = P_S = P_D = P_W = math.nan
            best = (pps[i].reshape(-1, 5)[np.argmax(pps[i].reshape(-1, 5)[:, 4])] if pps[i].size else pp_abf).tolist()
            extra = dict(fallback_abf=fb, p12=self.p12, p12_abf=self.p12_abf, W_abf=self.W_abf,
                         n_cs1=len(a["sets"]["cs"]), n_cs2=len(b["sets"]["cs"]),
                         PP_best=best, niter=[a["niter"], b["niter"]], fit_status=[a["status"], b["status"]],
                         converged=[a["converged"], b["converged"]])
            out.append(dict(locus_id=lc.locus_id, method=self.name, s=s, P_S=P_S, P_D=P_D, P_W=P_W,
                            E1=math.nan, E2=math.nan, status=status, runtime_s=rt, extra=json.dumps(extra)))
        return out

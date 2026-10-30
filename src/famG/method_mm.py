"""Family G-mm Method (INTERFACES §4): GPU port of coloc_susie_mm = susieR 0.16.6 susie_rss(z, R, n, L = 10,
R_finite = n_ref, R_mismatch = "eb_mix") per trait (famG.susie_mm) -> coloc.bf_bf over CS effects (p1 = p2 = 1e-4,
p12 = 5e-6) -> DESIGN §4.3 coloc-SuSiE adapter, coloc.abf fallback (W = 0.15, p12) when a trait has no CS.
Status as the wrapper (src/baselines/methods.py:ColocSusie): 'failed' when either susie_rss would stop with an error
(check_prior), else 'nonconverged' when either fit did not converge, else 'ok'."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch

from .coloc import bf_bf_batch, coloc_abf
from .method import _beta_se, _scalar_n, coloc_susie_adapter
from .susie_mm import MMConfig, susie_mm_batch


class GPUColocSusieMM:
    name = "famG_coloc_susie_mm_gpu"
    needs_ld = True

    def __init__(self, ld_dtype: str = "float32", p1=1e-4, p2=1e-4, p12=5e-6, W_abf=0.15, device=None,
                 max_batch: int = 1024, mem_budget_gb: float = 5.0, factor_ratio: float = 1.0, use_triton: bool = True,
                 streams: int = 1):
        """mem_budget_gb: total bucket-memory budget of this worker, split over `streams` concurrent buckets."""
        self.cfg = MMConfig(ld_dtype=getattr(torch, ld_dtype), max_batch=max_batch,
                            mem_budget_bytes=mem_budget_gb * 1e9 / max(1, streams), factor_ratio=factor_ratio,
                            use_triton=use_triton, streams=streams)
        self.p1, self.p2, self.p12, self.W_abf = p1, p2, p12, W_abf
        self.device = device

    def fit_batch(self, loci: list) -> list[dict]:
        t0 = time.time()
        B = len(loci)
        dicts = []
        for lc in loci:
            d = {"z": [np.asarray(lc.z1, np.float64), np.asarray(lc.z2, np.float64)],
                 "n": [_scalar_n(lc.n1), _scalar_n(lc.n2)], "n_ref": int(lc.n_ref)}
            if getattr(lc, "ld_type", "dense") == "factor":
                d["X"] = lc.X_ref
            else:
                d["R"] = lc.R
            dicts.append(d)
        fits = susie_mm_batch(dicts, self.cfg, device=self.device)

        def cs_rows(f):
            if f["status"] != "ok" or not f["sets"]["cs_index"]:
                return np.zeros((0, len(f["pip"])))
            return f["lbf_variable"][f["sets"]["cs_index"]]
        pps = bf_bf_batch([cs_rows(f[0]) for f in fits], [cs_rows(f[1]) for f in fits], self.p1, self.p2, self.p12,
                          device=self.device)
        rt = (time.time() - t0) / max(B, 1)
        out = []
        for i, lc in enumerate(loci):
            a, b = fits[i]
            if a["status"] != "ok" or b["status"] != "ok":
                status = "failed"
            elif not (a["converged"] and b["converged"]):
                status = "nonconverged"
            else:
                status = "ok"
            extra = dict(niter=[a["niter"], b["niter"]], converged=[a["converged"], b["converged"]],
                         fit_status=[a["status"], b["status"]], n_cs1=len(a["sets"]["cs"]), n_cs2=len(b["sets"]["cs"]),
                         lambda_bias=[a["lambda_bias"], b["lambda_bias"]], ld_dtype=str(self.cfg.ld_dtype))
            if status == "failed":
                extra["error"] = "Estimated prior variance is unreasonably large (susieR check_prior)"
                out.append(dict(locus_id=lc.locus_id, method=self.name, s=math.nan, P_S=math.nan, P_D=math.nan,
                                P_W=math.nan, E1=math.nan, E2=math.nan, status=status, runtime_s=rt,
                                extra=json.dumps(extra)))
                continue
            pp_abf = None
            if pps[i].size == 0:
                b1, s1 = _beta_se(lc.z1, lc.n1, getattr(lc, "maf1", None), getattr(lc, "beta1", None), getattr(lc, "se1", None))
                b2, s2 = _beta_se(lc.z2, lc.n2, getattr(lc, "maf2", None), getattr(lc, "beta2", None), getattr(lc, "se2", None))
                pp_abf = coloc_abf(b1, s1, b2, s2, self.p1, self.p2, self.p12, W=self.W_abf)
                extra["fallback_PP"] = list(map(float, pp_abf))
            else:
                extra["pairs_PP"] = pps[i].reshape(-1, 5).tolist()
            s, P_S, P_D, P_W, fb = coloc_susie_adapter(pps[i] if pps[i].size else None, pp_abf)
            extra["fallback"] = bool(fb)
            out.append(dict(locus_id=lc.locus_id, method=self.name, s=s, P_S=P_S, P_D=P_D, P_W=P_W,
                            E1=math.nan, E2=math.nan, status=status, runtime_s=rt, extra=json.dumps(extra)))
        return out

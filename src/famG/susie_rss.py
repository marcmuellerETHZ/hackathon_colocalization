"""Batched GPU SuSiE-RSS reproducing susieR 0.14.2 `susie_rss(z, R, n, L = 10)` (and the `n`-missing path).

Semantics reproduced from susieR 0.14.2 (susie_rss -> susie_suff_stat, see src/famG/README.md):
  * z is PVE-adjusted when n is given: z <- sqrt((n-1)/(z^2+n-2)) z; XtX = (n-1) R, Xty = sqrt(n-1) z,
    yty = n-1, standardize = TRUE, residual variance fixed at 1 (estimate_residual_variance = FALSE),
    scaled_prior_variance = 0.2 (V initialised to 0.2). Without n: XtX = R, Xty = z, n = 2, yty = 1,
    standardize = FALSE, V initialised to prior_variance = 50.
  * prior weights 1/p, with susieR's log(pi + sqrt(eps)) quirk;
  * per-effect prior variance: optim (exact batched port of R's Brent_fmin on log V in [-30, 15],
    keep-old-V rule, null check V <- 0), EM, or none;
  * check_prior: stop (here: status 'failed_prior_check') if any V > 100 * max|z|^2;
  * ELBO convergence (elbo[i+1] - elbo[i] < tol, tol = 1e-3), max_iter = 100;
  * credible sets exactly as susie_get_cs (V > 1e-9, de-duplication over all rows, purity = min |r|
    >= 0.5 from cov2cor(XtX), ordering by purity) with one documented exception: purity is computed
    over all CS members (susieR samples 100 members at random when a CS has > 100 members);
  * PIP = 1 - prod_l (1 - alpha_l) over effects with V > prior_tol (1e-9).

Numerics: everything that is not an LD product runs in float64; the LD products R v run in the LD
dtype (float32 by default, TF32 disabled and asserted; float64 available for exactness checks).
Loci are grouped into buckets of similar p (padding <= bucket_ratio, never to the global p_max);
padded variables get zero prior weight, hence exactly zero posterior mass.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .brent import brent_fmin
from .ld import BatchLD, build_batch_ld
from .precision import assert_no_tf32, check_ld_dtype, default_device

try:
    from .ser_kernel import optim_prior_variance
except Exception:  # pragma: no cover  (no triton)
    optim_prior_variance = None

SQRT_EPS = math.sqrt(2.220446049250313e-16)
M_LN_SQRT_2PI = 0.918938533204672741780329736406


@dataclass
class SusieRSSConfig:
    L: int = 10
    scaled_prior_variance: float = 0.2
    prior_variance_no_n: float = 50.0
    estimate_prior_variance: bool = True
    estimate_prior_method: str = "optim"   # optim | EM | none
    check_null_threshold: float = 0.0
    prior_tol: float = 1e-9
    max_iter: int = 100
    tol: float = 1e-3
    coverage: float = 0.95
    min_abs_corr: float = 0.5
    check_prior: bool = True
    ld_dtype: torch.dtype = torch.float32
    mem_budget_bytes: float = 6e9
    bucket_ratio: float = 1.10
    max_batch: int = 4096
    compact_fraction: float = 0.5
    return_mu: bool = False
    use_triton: bool = True       # fused Brent kernel on CUDA (src/famG/ser_kernel.py); False = pure torch port


def _dnorm_log(x: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    xs = x / sd
    return -(M_LN_SQRT_2PI + 0.5 * xs * xs + torch.log(sd))


def _lbf(betahat, shat2, V):
    """susieR: dnorm(betahat, 0, sqrt(V + shat2), log=TRUE) - dnorm(betahat, 0, sqrt(shat2), log=TRUE)."""
    return _dnorm_log(betahat, torch.sqrt(V + shat2)) - _dnorm_log(betahat, torch.sqrt(shat2))


def _logsum_r(lpo: torch.Tensor):
    """R: maxlpo + log(sum(exp(lpo - maxlpo))) along the last dim. Returns (value, w, sumw, maxlpo)."""
    m = lpo.max(dim=-1, keepdim=True).values
    w = torch.exp(lpo - m)
    s = w.sum(-1)
    return torch.log(s) + m[..., 0], w, s, m[..., 0]


class _Bucket:
    """State of one bucket of loci of (padded) width P, all on device."""

    def __init__(self, z, valid, p_real, n, ld: BatchLD, cfg: SusieRSSConfig):
        self.cfg = cfg
        dev = z.device
        f64 = torch.float64
        B, P = z.shape
        L = cfg.L
        self.ld = ld
        self.valid = valid
        self.n_mode = n is not None
        dR = ld.diag()
        if self.n_mode:
            nm1 = (n - 1.0)[:, None]
            adj = nm1 / (z * z + n[:, None] - 2.0)
            zt = torch.sqrt(adj) * z
            Xty = torch.sqrt(nm1) * zt
            dXtX = nm1 * dR
            csd = torch.sqrt(dXtX / nm1)
            csd = torch.where(csd == 0, torch.ones_like(csd), csd)
            d = ((1.0 / csd) * dXtX) / csd
            Xty = Xty / csd
            self.nm1 = nm1[:, 0]
            self.yty = self.nm1.clone()
            self.n_elbo = n.clone()
            Vinit = cfg.scaled_prior_variance * (self.yty / self.nm1)
            self.sigma2 = self.yty / self.nm1
        else:
            Xty = z.clone()
            csd = torch.ones_like(z)
            d = dR.clone()
            self.nm1 = None
            self.yty = torch.ones(B, dtype=f64, device=dev)
            self.n_elbo = torch.full((B,), 2.0, dtype=f64, device=dev)
            Vinit = cfg.prior_variance_no_n * self.yty  # varY = yty/(n-1) = 1
            self.sigma2 = torch.ones(B, dtype=f64, device=dev)
        d = torch.where(valid, d, torch.ones_like(d))
        Xty = torch.where(valid, Xty, torch.zeros_like(Xty))
        self.csd, self.d, self.Xty = csd, d, Xty
        pw = 1.0 / p_real.to(f64)
        self.lpw = torch.where(valid, torch.log(pw[:, None] + SQRT_EPS).expand(B, P), torch.full_like(z, -math.inf))
        # initial state (init_setup): alpha = 1/p, mu = mu2 = 0
        self.alpha = torch.where(valid[:, None, :], (pw[:, None, None]).expand(B, L, P), torch.zeros(1, dtype=f64, device=dev)).clone()
        self.mu = torch.zeros((B, L, P), dtype=f64, device=dev)
        self.mu2 = torch.zeros((B, L, P), dtype=f64, device=dev)
        self.V = Vinit[:, None].expand(B, L).clone()
        self.KL = torch.full((B, L), math.nan, dtype=f64, device=dev)
        self.lbf = torch.full((B, L), math.nan, dtype=f64, device=dev)
        self.lbf_variable = torch.full((B, L, P), math.nan, dtype=f64, device=dev)
        self.XtXr = torch.zeros((B, P), dtype=f64, device=dev)
        self.XtXb = torch.zeros((B, L, P), dtype=f64, device=dev)
        bhat = Xty / d
        shat = torch.sqrt(self.sigma2[:, None] / d)
        zz = torch.where(valid, (bhat / shat).abs(), torch.zeros_like(bhat))
        self.zm = zz.max(1).values
        self.elbo_prev = torch.full((B,), -math.inf, dtype=f64, device=dev)
        self.elbo_hist = [[] for _ in range(B)]
        self.niter = torch.zeros(B, dtype=torch.int64, device=dev)
        self.converged = torch.zeros(B, dtype=torch.bool, device=dev)
        self.status = ["ok"] * B
        self.active = torch.ones(B, dtype=torch.bool, device=dev)
        self.orig = torch.arange(B, device=dev)

    # ---- LD product with susieR's standardisation ---------------------------------------
    def XtX_mv(self, v: torch.Tensor) -> torch.Tensor:
        """XtX_std v for v (B,P) or (B,K,P)."""
        if self.n_mode:
            if v.dim() == 2:
                return self.nm1[:, None] * self.ld.matvec(v / self.csd) / self.csd
            return self.nm1[:, None, None] * self.ld.matvec(v / self.csd[:, None, :]) / self.csd[:, None, :]
        return self.ld.matvec(v)

    # ---- single effect regression (batched over loci) ------------------------------------
    def _loglik(self, V, betahat, shat2):
        lbf = _lbf(betahat, shat2, V[:, None])
        val, _, _, _ = _logsum_r(lbf + self.lpw)
        return val

    def ser(self, XtR, Vold):
        cfg = self.cfg
        d, s2 = self.d, self.sigma2[:, None]
        betahat = (1.0 / d) * XtR
        shat2 = s2 / d
        method = cfg.estimate_prior_method if cfg.estimate_prior_variance else "none"
        B = XtR.shape[0]
        if method == "optim":
            if XtR.is_cuda and cfg.use_triton and optim_prior_variance is not None:
                V = optim_prior_variance(betahat, shat2, self.lpw, Vold, self.active, cfg.check_null_threshold)
            else:
                f = lambda lV: -self._loglik(torch.exp(lV), betahat, shat2)
                lV = brent_fmin(f, -30.0, 15.0, B, XtR.device)
                keep = f(lV) > f(torch.log(Vold))
                lV = torch.where(keep, torch.log(Vold), lV)
                V = torch.exp(lV)
                V = self._null_check(V, betahat, shat2)
        else:
            V = Vold
        lbf = _lbf(betahat, shat2, V[:, None])
        lpo = lbf + self.lpw
        lbf_model, w, sw, _ = _logsum_r(lpo)
        alpha = w / sw[:, None]
        post_var = 1.0 / (1.0 / V[:, None] + d / s2)
        post_mean = (1.0 / s2) * post_var * XtR
        post_mean2 = post_var + post_mean * post_mean
        if method == "EM":
            Vn = (alpha * post_mean2).sum(1)
            V = self._null_check(Vn, betahat, shat2)
        Eb, Eb2 = alpha * post_mean, alpha * post_mean2
        e_loglik = -0.5 / s2[:, 0] * (-2.0 * (Eb * XtR).sum(1) + (d * Eb2).sum(1))
        KL = -lbf_model + e_loglik
        lbf = torch.where(self.valid, lbf, torch.zeros_like(lbf))
        return alpha, post_mean, post_mean2, lbf, V, lbf_model, KL

    def _null_check(self, V, betahat, shat2):
        l0 = self._loglik(torch.zeros_like(V), betahat, shat2)
        lV = self._loglik(V, betahat, shat2)
        return torch.where(l0 + self.cfg.check_null_threshold >= lV, torch.zeros_like(V), V)

    # ---- one IBSS sweep -----------------------------------------------------------------
    def sweep(self):
        act = self.active
        a3 = act[:, None]
        for l in range(self.cfg.L):
            XtXr = self.XtXr - self.XtXb[:, l]
            XtR = self.Xty - XtXr
            alpha, mu, mu2, lbf, V, lbf_model, KL = self.ser(XtR, self.V[:, l])
            b = alpha * mu
            XtXb = self.XtX_mv(b)
            self.alpha[:, l] = torch.where(a3, alpha, self.alpha[:, l])
            self.mu[:, l] = torch.where(a3, mu, self.mu[:, l])
            self.mu2[:, l] = torch.where(a3, mu2, self.mu2[:, l])
            self.lbf_variable[:, l] = torch.where(a3, lbf, self.lbf_variable[:, l])
            self.V[:, l] = torch.where(act, V, self.V[:, l])
            self.lbf[:, l] = torch.where(act, lbf_model, self.lbf[:, l])
            self.KL[:, l] = torch.where(act, KL, self.KL[:, l])
            self.XtXb[:, l] = torch.where(a3, XtXb, self.XtXb[:, l])
            self.XtXr = torch.where(a3, XtXr + XtXb, self.XtXr)

    def elbo(self):
        Bm = self.alpha * self.mu
        XB2 = (Bm * self.XtXb).sum((1, 2))
        betabar = Bm.sum(1)
        postb2 = (self.alpha * self.mu2).sum(1)
        ER2 = (self.yty - 2.0 * (betabar * self.Xty).sum(1) + (betabar * self.XtXr).sum(1) - XB2
               + (self.d * postb2 * self.valid).sum(1))
        s2 = self.sigma2
        eloglik = -self.n_elbo / 2.0 * torch.log(2.0 * math.pi * s2) - 1.0 / (2.0 * s2) * ER2
        return eloglik - self.KL.sum(1)

    def select(self, idx):
        for name in ["valid", "csd", "d", "Xty", "lpw", "alpha", "mu", "mu2", "V", "KL", "lbf", "lbf_variable",
                     "XtXr", "XtXb", "zm", "elbo_prev", "niter", "converged", "active", "orig", "sigma2", "yty",
                     "n_elbo"]:
            setattr(self, name, getattr(self, name)[idx])
        if self.nm1 is not None:
            self.nm1 = self.nm1[idx]
        il = idx.tolist()
        self.elbo_hist = [self.elbo_hist[i] for i in il]
        self.status = [self.status[i] for i in il]
        self.ld = self.ld.select(idx)


def _credible_sets(alpha: np.ndarray, V: np.ndarray, ld: BatchLD, b: int, p: int, cfg: SusieRSSConfig):
    """susie_get_cs (susieR 0.14.2) for one locus. alpha: (L,p) float64 numpy; V: (L,) numpy.

    Purity is computed on all members (susieR samples n_purity = 100 members of larger sets)."""
    L = alpha.shape[0]
    order = np.argsort(-alpha, axis=1, kind="stable")
    srt = np.take_along_axis(alpha, order, axis=1)
    cum = np.cumsum(srt, axis=1)
    n_in = np.minimum((cum < cfg.coverage).sum(1) + 1, p)
    cs_all = [tuple(sorted(order[l, : n_in[l]].tolist())) for l in range(L)]
    cov_all = [float(srt[l, : n_in[l]].sum()) for l in range(L)]
    seen, inc = set(), []
    for l in range(L):
        dup = cs_all[l] in seen
        seen.add(cs_all[l])
        inc.append(bool(V[l] > 1e-9) and len(cs_all[l]) > 0 and not dup)
    idx_inc = [l for l in range(L) if inc[l]]
    empty = {"cs": [], "cs_index": [], "purity": [], "coverage": []}
    if not idx_inc:
        return empty
    pur = []
    for l in idx_inc:
        cs = cs_all[l]
        if len(cs) == 1:
            pur.append((1.0, 1.0, 1.0))
            continue
        ix = torch.as_tensor(cs, device=ld.mat.device)
        m = len(cs)
        if m > 1024:
            # large CS: min |r| in row chunks first (bounded memory); only a pure set needs the full matrix
            vmin = math.inf
            for c0 in range(0, m, 512):
                Cc = ld.corr_rows(b, ix, c0, min(m, c0 + 512)).abs()          # rows c0.. of corr[ix, ix]
                rr = torch.arange(c0, min(m, c0 + 512), device=Cc.device)[:, None]
                cc = torch.arange(m, device=Cc.device)[None, :]
                vmin = min(vmin, float(torch.where(cc > rr, Cc, torch.full_like(Cc, math.inf)).min()))
                if vmin < cfg.min_abs_corr:
                    break
            if vmin < cfg.min_abs_corr:
                pur.append((vmin, math.nan, math.nan))
                continue
        C = ld.corr_sub(b, ix).abs()
        iu = torch.triu_indices(m, m, offset=1, device=C.device)
        vals = C[iu[0], iu[1]]
        vmin = float(vals.min())
        if vmin < cfg.min_abs_corr:   # impure: dropped, mean/median not needed
            pur.append((vmin, math.nan, math.nan))
            continue
        vs, _ = torch.sort(vals)
        k = vs.numel()
        med = float(vs[k // 2]) if k % 2 == 1 else float(0.5 * (vs[k // 2 - 1] + vs[k // 2]))
        pur.append((vmin, float(vals.sum() / k), med))
    pmin = np.array([q[0] for q in pur])
    is_pure = [i for i in range(len(idx_inc)) if pmin[i] >= cfg.min_abs_corr]
    if not is_pure:
        return empty
    ordering = sorted(range(len(is_pure)), key=lambda k: -pmin[is_pure[k]])  # stable, decreasing
    out = {"cs": [], "cs_index": [], "purity": [], "coverage": []}
    for k in ordering:
        i = is_pure[k]
        l = idx_inc[i]
        out["cs"].append(list(cs_all[l]))
        out["cs_index"].append(l)
        out["purity"].append(pur[i])
        out["coverage"].append(cov_all[l])
    return out


def _fit_bucket(zs, lds, ns, kind, P, cfg, device, n_refs=None):
    B = len(zs)
    f64 = torch.float64
    z = torch.zeros((B, P), dtype=f64, device=device)
    valid = torch.zeros((B, P), dtype=torch.bool, device=device)
    p_real = torch.zeros(B, dtype=torch.int64, device=device)
    for i, zi in enumerate(zs):
        p = len(zi)
        zz = np.nan_to_num(np.asarray(zi, dtype=np.float64), nan=0.0)  # susie_rss: z[is.na(z)] = 0
        z[i, :p] = torch.as_tensor(zz, device=device)
        valid[i, :p] = True
        p_real[i] = p
    n = None if ns[0] is None else torch.as_tensor(np.asarray(ns, dtype=np.float64), device=device)
    ld = build_batch_ld(lds, kind, P, cfg.ld_dtype, device)
    if kind == "factor" and n_refs is not None:
        ld.n_ref = torch.as_tensor(np.asarray(n_refs, dtype=np.float64), device=device)
    st = _Bucket(z, valid, p_real, n, ld, cfg)
    p_real_np = p_real.cpu().numpy()
    results = [None] * B
    finished = {}

    def flush(rows):
        if not rows:
            return
        ri = torch.as_tensor(rows, device=device)
        A = st.alpha[ri].cpu().numpy()
        Vv = st.V[ri].cpu().numpy()
        LBF = st.lbf_variable[ri].cpu().numpy()
        lbf_m = st.lbf[ri].cpu().numpy()
        KLm = st.KL[ri].cpu().numpy()
        nit = st.niter[ri].cpu().numpy()
        conv = st.converged[ri].cpu().numpy()
        orig = st.orig[ri].cpu().numpy()
        if cfg.return_mu:
            MU = st.mu[ri].cpu().numpy()
            MU2 = st.mu2[ri].cpu().numpy()
        for k, r in enumerate(rows):
            o = int(orig[k])
            p = int(p_real_np[o])
            alpha = A[k, :, :p]
            V = Vv[k]
            sets = _credible_sets(alpha, V, st.ld, r, p, cfg)
            keep = V > cfg.prior_tol
            pip = 1.0 - np.prod(1.0 - alpha[keep], axis=0) if keep.any() else np.zeros(p)
            res = {
                "pip": pip, "alpha": alpha, "lbf_variable": LBF[k, :, :p], "V": V,
                "lbf": lbf_m[k], "KL": KLm[k], "elbo": np.array(st.elbo_hist[r]), "niter": int(nit[k]),
                "converged": bool(conv[k]), "status": st.status[r], "sets": sets,
            }
            if cfg.return_mu:
                res["mu"] = MU[k, :, :p]
                res["mu2"] = MU2[k, :, :p]
            results[o] = res

    for it in range(1, cfg.max_iter + 1):
        st.sweep()
        act = st.active
        if cfg.check_prior:
            bad = act & (st.V > 100.0 * (st.zm[:, None] ** 2)).any(1)
            if bool(bad.any()):
                for r in torch.nonzero(bad).flatten().tolist():
                    st.status[r] = "failed_prior_check"
                act = act & ~bad
        el = st.elbo()
        st.niter = torch.where(act, torch.full_like(st.niter, it), st.niter)
        el_c = el.cpu().tolist()
        actl = act.cpu().tolist()
        for r in range(len(actl)):
            if actl[r]:
                st.elbo_hist[r].append(el_c[r])
                if not math.isfinite(el_c[r]):
                    st.status[r] = "failed_infinite_elbo"
        conv = act & ((el - st.elbo_prev) < cfg.tol)
        st.converged = st.converged | conv
        st.elbo_prev = torch.where(act, el, st.elbo_prev)
        fail = torch.tensor([s != "ok" for s in st.status], device=device)
        st.active = act & ~conv & ~fail
        n_act = int(st.active.sum())
        if n_act == 0:
            break
        if n_act < cfg.compact_fraction * st.active.numel():
            done = torch.nonzero(~st.active).flatten().tolist()
            flush(done)
            st.select(torch.nonzero(st.active).flatten())
    flush([r for r in range(st.active.numel()) if results[int(st.orig[r])] is None])
    return results


def susie_rss_batch(zs: list, lds: list, ns: list | None = None, ld_kind: str | list = "dense",
                    cfg: SusieRSSConfig | None = None, device=None, n_refs: list | None = None,
                    verbose: bool = False) -> list[dict]:
    """Fit SuSiE-RSS to many loci at once.

    zs: list of z vectors (p_i,); lds: list of LD inputs: dense R (p_i,p_i) or factor X (n_ref_i,p_i);
    ns: list of sample sizes (None entries = susie_rss without n); ld_kind: 'dense'/'factor' (or list).
    Returns a list of result dicts in input order.
    """
    cfg = cfg or SusieRSSConfig()
    check_ld_dtype(cfg.ld_dtype)
    device = device or default_device()
    if device.type == "cuda" if isinstance(device, torch.device) else str(device).startswith("cuda"):
        assert_no_tf32()
    N = len(zs)
    ns = ns if ns is not None else [None] * N
    kinds = ld_kind if isinstance(ld_kind, list) else [ld_kind] * N
    n_refs = n_refs if n_refs is not None else [None] * N
    out = [None] * N
    bytes_per = 4 if cfg.ld_dtype == torch.float32 else 8
    groups = {}
    for i in range(N):
        key = (kinds[i], ns[i] is None)
        groups.setdefault(key, []).append(i)
    for (kind, no_n), idxs in groups.items():
        idxs = sorted(idxs, key=lambda i: len(zs[i]))
        bucket = []

        def run(bucket):
            if not bucket:
                return
            P = max(len(zs[i]) for i in bucket)
            t0 = time.time()
            res = _fit_bucket([zs[i] for i in bucket], [lds[i] for i in bucket],
                              [ns[i] for i in bucket], kind, P, cfg, device,
                              n_refs=[n_refs[i] if n_refs[i] is not None else lds[i].shape[0] for i in bucket]
                              if kind == "factor" else None)
            if verbose:
                print(f"bucket kind={kind} B={len(bucket)} P={P} {time.time() - t0:.2f}s", flush=True)
            for i, r in zip(bucket, res):
                out[i] = r

        p_min = None
        for i in idxs:
            p = len(zs[i])
            if kind == "dense":
                per = p * p * bytes_per + 64 * cfg.L * p * 8
            else:
                per = lds[i].shape[0] * p * bytes_per + 64 * cfg.L * p * 8
            if bucket and (p > cfg.bucket_ratio * p_min or (len(bucket) + 1) * per > cfg.mem_budget_bytes
                           or len(bucket) >= cfg.max_batch):
                run(bucket)
                bucket = []
            if not bucket:
                p_min = p
            bucket.append(i)
        run(bucket)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out

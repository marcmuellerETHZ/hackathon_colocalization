"""G-mm: batched GPU port of susieR 0.16.6 `susie_rss(z, R, n, L = 10, R_finite = n_ref, R_mismatch = "eb_mix")`,
the call made by the coloc_susie_mm baseline (src/baselines/methods.py:ColocSusie(mismatch=True), worker.R fit_susie).

Every step below is a line-by-line port of susieR 0.16.6 (env/src/susieR, commit 97feba5); R file:function in [].

Setup [susie_constructors.R: summary_stats_constructor, sufficient_stats_constructor]
  * z <- sqrt((n-1)/(z^2+n-2)) z (PVE adjustment, z_method = "wald"); data$z_marginal = adjusted z.
  * XtX = (n-1) R, Xty = sqrt(n-1) z, yty = n-1; standardize: csd = sqrt(diag(XtX)/(n-1)), XtX <- XtX/(csd csd'),
    Xty <- Xty/csd; predictor weights d = diag(XtX). sigma2 = var_y = 1 (estimate_residual_variance = FALSE).
  * V initialised to scaled_prior_variance * var_y = 0.2; alpha = 1/p, mu = mu2 = 0; pi = 1/p,
    log prior weights log(pi + sqrt(eps)) [susie_utils.R: lbf_stabilization].
  * max_iter = 50 (susie_rss default for summary statistics), tol = 1e-4, convergence_method forced to "pip"
    [force_pip], check_prior = TRUE [susie_rss_control].
Initialisation [rss_mismatch.R: initialize_R_mismatch]
  * lambda = 0; one single_effect_update of effect 1; lambda <- estimate_lambda_bias(fit residual);
    lambda <- lambda * ld_coherence(alpha_1) [compute_ser_ld_coherence]. The fitted effect 1 is kept (warm start).
IBSS iteration [iterative_bayesian_stepwise_selection.R: ibss_fit; susie_workhorse.R]
  * for l = 1..L: residual r = Xty - (XtXr - XtX alpha_l mu_l); per-variable inflation
      s_j = (XtXr_{-l})_j^2/(n-1) + max(b_{-l}' XtXr_{-l}, 0),
      infl_j = 1 + (1/B + w_j lambda) s_j / sigma2   (eb_mix, lambda > 0; w_j = mixture gate)
      infl_j = 1 + (1/B + lambda) s_j / sigma2        (lambda == 0)        [rss_mismatch.R: compute_shat2_inflation]
    SER with betahat = r/d, shat2 = sigma2/d * infl; prior variance by optim(Brent) on log V in [-30, 15] with
    the keep-old rule and the null check [single_effect_regression.R]; closed-form Gaussian lbf.
  * after the sweep: lambda <- estimate_lambda_bias (MLE, optimize tol 1e-8, nll(0) check, 0.1 * SE_0 floor).
  * validate_prior: any V > 100 max|z|^2 -> R error "Estimated prior variance is unreasonably large" (status failed).
  * convergence [susie_utils.R: check_alpha_pip_cycle_convergence]: iteration 1 never converges; afterwards
    converged if max(|d alpha|, |d PIP|) < tol at lag 1, or at lag 2..5 (short cycle: alpha <- mean over the cycle);
    then the lambda guard: not converged if |d lambda| > tol (the cycle-averaged alpha is kept).
Finalisation: trim effects with V < prior_tol; credible sets as susie_get_cs (exact purity, cov2cor of XtX);
PIP over effects with V > prior_tol.

Skipped, with reasons (no effect on any returned quantity):
  * eigen(R), Q_art, artifact_flag, R_reliability_flag, BF attenuation, the one-SER ser_model: diagnostics only.
  * ELBO: under PIP convergence it only matters through get_objective's is.infinite() stop; on the ss path
    compute_augmented_eloglik_ss reads data$predictor_weights, which is NULL, so the data term is the constant
    -p/2 log(2 pi) and the ELBO is always finite.

Numerics: all arithmetic except LD products in float64; LD products in `ld_dtype` (float64 for identity tests,
float32 by default for speed; TF32 disabled and asserted). The three 1-D optimisations use exact ports of R's
Brent_fmin (torch: famG.brent.brent_fmin; CUDA: fused Triton kernels in famG.mm_kernels).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .brent import R_OPTIM_RELTOL, brent_fmin
from .ld import BatchLD, build_batch_ld
from .precision import assert_no_tf32, check_ld_dtype, default_device
from .susie_rss import _credible_sets

try:
    from . import mm_kernels as K
except Exception:  # pragma: no cover  (no triton)
    K = None

DBL_EPS = 2.220446049250313e-16
SQRT_EPS = math.sqrt(DBL_EPS)
DBL_MAX = 1.7976931348623157e308
M_LN_SQRT_2PI = 0.918938533204672741780329736406
LOG_2PI = math.log(2 * math.pi)


def qnorm_upper(p: float) -> float:
    """qnorm(p, lower.tail = FALSE) (scipy; for the eb_mix reference z, 5.451310... at p = 2.5e-8)."""
    from scipy.stats import norm
    return float(norm.isf(p))


@dataclass
class MMConfig:
    L: int = 10
    scaled_prior_variance: float = 0.2
    max_iter: int = 50
    tol: float = 1e-4
    prior_tol: float = 1e-9
    check_null_threshold: float = 0.0
    coverage: float = 0.95
    min_abs_corr: float = 0.5
    check_prior: bool = True
    eb_mix_ref_p: float = 5e-8
    cycle_window: int = 5
    ld_dtype: torch.dtype = torch.float32
    factor_ratio: float = 1.0        # factor LD X (n_ref x p) used when n_ref < factor_ratio * p, else dense R
    mem_budget_bytes: float = 5e9
    bucket_ratio: float = 1.5
    max_batch: int = 1024            # loci per bucket
    compact_fraction: float = 0.5
    use_triton: bool = True
    streams: int = 1                 # buckets fitted concurrently on separate CUDA streams (threads); tails overlap


# ---------------------------------------------------------------------------------------------- helpers
def _dnorm_log(x, sd):
    xs = x / sd
    return -(M_LN_SQRT_2PI + 0.5 * xs * xs + torch.log(sd))


def _plogis(x):
    """R's plogis(x) = 1 / (1 + exp(-x)) (exp overflow -> exactly 0, as in R)."""
    return 1.0 / (1.0 + torch.exp(-x))


def _ser_lbf(betahat, shat2, V):
    """gaussian_ser_lbf (susieR 0.16.6)."""
    s2 = torch.clamp(shat2, min=DBL_EPS)
    return -0.5 * torch.log(1.0 + V / s2) + 0.5 * betahat * betahat * V / (s2 * (V + s2))


def _logsum2(lpo):
    """R: log(sum(exp(lpo - max(lpo)))) + max(lpo) along the last dim (two-pass). Returns (value, w, sumw)."""
    m = lpo.max(dim=-1, keepdim=True).values
    w = torch.exp(lpo - m)
    s = w.sum(-1)
    return torch.log(s) + m[..., 0], w, s


def _quantile7(x, n, prob):
    """R quantile(x, prob) type 7 per row over the first n[r] entries of x (rows of x padded with anything).
    x: (B,P) float64; n: (B,) int64. Uses (1-h) x[lo] + h x[hi] as R does. Only the top order statistics are
    needed (prob near 1), so they come from topk: ascending x[k] (1-based) = descending x[n - k] (0-based)."""
    B, P = x.shape
    ar = torch.arange(P, device=x.device)
    xs = torch.where(ar[None, :] < n[:, None], x, torch.full_like(x, -math.inf))
    index = 1.0 + (n.to(torch.float64) - 1.0).clamp(min=0) * prob   # 1-based
    lo = torch.floor(index).to(torch.int64)
    hi = torch.ceil(index).to(torch.int64)
    kmax = int(math.ceil(0.01 * P)) + 2 if prob == 0.99 else P      # n - lo <= ceil(0.01 (n-1)) + 1 (no host sync)
    desc = torch.topk(xs, min(P, kmax), dim=1, sorted=True).values
    xlo = desc.gather(1, (n - lo)[:, None])[:, 0]
    xhi = desc.gather(1, (n - hi)[:, None])[:, 0]
    h = index - lo.to(torch.float64)
    upd = (index > lo.to(torch.float64)) & (xhi != xlo)
    return torch.where(upd, (1.0 - h) * xlo + h * xhi, xlo)


# ---------------------------------------------------------------------------------------------- bucket state
class MMBucket:
    """B loci x T traits (rows = B*T fits, row = b*T + t) sharing one LD per locus."""

    def __init__(self, z, valid, p_real, n, B_ref, ld: BatchLD, T: int, cfg: MMConfig):
        self.cfg, self.T, self.ld = cfg, T, ld
        dev = z.device
        f64 = torch.float64
        NR, P = z.shape
        L = cfg.L
        self.valid, self.p_real = valid, p_real                       # (NR,P), (NR,)
        self.Bref = B_ref.to(f64)                                      # (NR,)
        nm1 = (n - 1.0)
        self.nm1 = nm1
        zt = torch.sqrt(nm1[:, None] / (z * z + n[:, None] - 2.0)) * z
        zt = torch.where(valid, zt, torch.zeros_like(zt))
        self.z_marg = zt
        dR = ld.diag().repeat_interleave(T, 0)                         # (NR,P)
        dXtX = nm1[:, None] * dR
        csd = torch.sqrt(dXtX / nm1[:, None])
        csd = torch.where(csd == 0, torch.ones_like(csd), csd)
        csd = torch.where(valid, csd, torch.ones_like(csd))
        d = ((1.0 / csd) * dXtX) / csd
        d = torch.where(valid, d, torch.ones_like(d))
        Xty = torch.sqrt(nm1)[:, None] * zt / csd
        self.csd, self.d, self.Xty = csd, d, torch.where(valid, Xty, torch.zeros_like(Xty))
        self.sigma2 = torch.ones(NR, dtype=f64, device=dev)            # yty/(n-1)
        pw = 1.0 / p_real.to(f64)
        self.pi = pw
        self.lpw = torch.where(valid, torch.log(pw[:, None] + SQRT_EPS).expand(NR, P), torch.full_like(z, -math.inf))
        self.alpha = torch.where(valid[:, None, :], pw[:, None, None].expand(NR, L, P),
                                 torch.zeros(1, dtype=f64, device=dev)).clone()
        self.mu = torch.zeros((NR, L, P), dtype=f64, device=dev)
        self.mu2 = torch.zeros((NR, L, P), dtype=f64, device=dev)
        self.V = torch.full((NR, L), cfg.scaled_prior_variance, dtype=f64, device=dev)
        self.lbf = torch.full((NR, L), math.nan, dtype=f64, device=dev)
        self.lbf_variable = torch.full((NR, L, P), math.nan, dtype=f64, device=dev)
        self.XtXr = torch.zeros((NR, P), dtype=f64, device=dev)
        self.XtXb = torch.zeros((NR, L, P), dtype=f64, device=dev)    # cache of XtX (alpha_l mu_l)
        self.lam = torch.zeros(NR, dtype=f64, device=dev)
        self.lam_diff = torch.zeros(NR, dtype=f64, device=dev)
        # check_prior: zm = max |Xty/d / sqrt(sigma2/d)|
        zz = (self.Xty / d) / torch.sqrt(self.sigma2[:, None] / d)
        self.zm = torch.where(valid, zz.abs(), torch.zeros_like(zz)).max(1).values
        # eb_mix prior log-odds: 0.5 (max z_marginal^2 - z_ref^2); pi_signal = plogis(-lpo)
        zref = qnorm_upper(cfg.eb_mix_ref_p / 2.0)
        z2max = torch.where(valid, zt * zt, torch.zeros_like(zt)).max(1).values
        self.gate_lpo = 0.5 * (z2max - zref * zref)
        self.pi_signal = _plogis(-self.gate_lpo)
        self.niter = torch.zeros(NR, dtype=torch.int64, device=dev)
        self.converged = torch.zeros(NR, dtype=torch.bool, device=dev)
        self.failed = torch.zeros(NR, dtype=torch.bool, device=dev)
        self.active = torch.ones(NR, dtype=torch.bool, device=dev)
        self.orig = torch.arange(NR, device=dev)
        self.hist_alpha: list[torch.Tensor] = []
        self.hist_pip: list[torch.Tensor] = []
        self.prev_alpha = None
        self.coherence = torch.ones(NR, dtype=f64, device=dev)
        self.lam_init = torch.zeros(NR, dtype=f64, device=dev)
        self.use_k = bool(cfg.use_triton and K is not None and z.is_cuda)

    # ---- LD -------------------------------------------------------------------------------------------
    def XtX_mv(self, v):
        """standardised XtX v for v (NR,P) or (NR,K,P); the T traits of a locus share one LD product."""
        sq = v.dim() == 2
        vv = v.unsqueeze(1) if sq else v                                  # (NR,K,P)
        NR, Kk, P = vv.shape
        u = vv / self.csd[:, None, :]
        u = u.reshape(NR // self.T, self.T * Kk, P)
        if self.use_k and self.ld.mat.dtype == torch.float64:
            # fp64 LD: memory-bound Triton GEMV (cuBLAS fp64 GEMM pads the 2-row operand to a full tile)
            if self.ld.kind == "dense":
                out = K.vtm(u, self.ld.mat)                                     # u R (R symmetric)
            else:
                t = K.mv(u, self.ld.mat)                                        # X u'  (B,TK,n)
                out = K.vtm(t, self.ld.mat) / self.ld.n_ref[:, None, None]      # X' X u' / n_ref
            out = out.reshape(NR, Kk, P)
        else:
            out = self.ld.matvec(u).reshape(NR, Kk, P)
        out = self.nm1[:, None, None] * out / self.csd[:, None, :]
        out = torch.where(self.valid[:, None, :], out, torch.zeros_like(out))
        return out[:, 0] if sq else out

    # ---- 1-D optimisations ---------------------------------------------------------------------------
    def _loglik(self, V, betahat, shat2):
        lbf = _ser_lbf(betahat, shat2, V[:, None])
        val, _, _ = _logsum2(lbf + self.lpw)
        return val

    def _optim_V(self, betahat, shat2, Vold, rows_mask=None):
        nk = self.p_real if rows_mask is None else torch.where(rows_mask, self.p_real, torch.zeros_like(self.p_real))
        nk = nk.to(torch.int32)                     # inactive rows are skipped inside the kernels (results unused)
        if self.use_k:
            lV = K.brent_ser(betahat, shat2, self.lpw, nk)
        else:
            f = lambda x: -self._loglik(torch.exp(x), betahat, shat2)
            lV = brent_fmin(f, -30.0, 15.0, betahat.shape[0], betahat.device, tol=R_OPTIM_RELTOL)
        Vopt = torch.exp(lV)
        Vold_ = torch.exp(torch.log(Vold))
        if self.use_k:
            ll = K.loglik3(betahat, shat2, self.lpw, torch.stack([Vopt, Vold_, torch.zeros_like(Vopt)], 1), nk)
            f_opt, f_old, l0 = -ll[:, 0], -ll[:, 1], ll[:, 2]
        else:
            f_opt = -self._loglik(Vopt, betahat, shat2)
            f_old = -self._loglik(Vold_, betahat, shat2)
            l0 = self._loglik(torch.zeros_like(Vopt), betahat, shat2)
        keep_old = f_opt > f_old
        V = torch.where(keep_old, Vold, Vopt)
        l1 = torch.where(keep_old, -f_old, -f_opt)      # loglik(V) (V is one of the two evaluated points)
        return torch.where(l0 + self.cfg.check_null_threshold >= l1, torch.zeros_like(V), V)

    def _vsig(self, rz, tau0, pi):
        """estimate_eb_mix_vsig for rows (all valid entries finite and tau0 > 0 by construction)."""
        valid = self.valid_sub
        n = self.p_sub
        neg = torch.full_like(rz, -math.inf)
        rz2 = rz * rz
        rz2max = torch.where(valid, rz2, neg).max(1).values
        tmax = torch.where(valid, tau0, neg).max(1).values
        tmin = torch.where(valid, tau0, torch.full_like(tau0, math.inf)).min(1).values
        q99 = _quantile7(rz2, n, 0.99)
        one = torch.ones_like(q99)
        scale_v = torch.stack([one, q99, tmax, rz2max], 1).max(1).values
        lower_v = torch.clamp(tmin * 0.01, min=DBL_EPS)
        upper_v = torch.stack([one, 10 * scale_v, 100 * rz2max, 100 * tmax], 1).max(1).values
        pi0 = pi <= 0
        degenerate = (~torch.isfinite(lower_v)) | (~torch.isfinite(upper_v)) | (lower_v >= upper_v)
        lo, hi = torch.log(lower_v), torch.log(upper_v)
        lo = torch.where(degenerate, torch.zeros_like(lo), lo)
        hi = torch.where(degenerate, torch.ones_like(hi), hi)
        pis = torch.where(pi0, torch.full_like(pi, 0.5), pi)
        log0 = torch.log1p(-pis)[:, None] + _dnorm_log(rz, torch.sqrt(tau0))
        log0 = torch.where(valid, log0, neg)
        if self.use_k:
            nk = n if self.gate_need is None else torch.where(self.gate_need, n, torch.zeros_like(n))
            x = K.brent_vsig(rz, log0, torch.log(pis), nk, lo, hi)
        else:
            lpis = torch.log(pis)

            def nll(lv):
                Vv = torch.exp(lv)
                log1 = lpis[:, None] + _dnorm_log(rz, torch.sqrt(Vv)[:, None])
                m = torch.maximum(log0, log1)
                t = m + torch.log(torch.exp(log0 - m) + torch.exp(log1 - m))
                return -torch.where(valid, t, torch.zeros_like(t)).sum(1)
            x = brent_fmin(nll, lo, hi, rz.shape[0], rz.device, tol=1e-6)
        Vs = torch.exp(x)
        Vs = torch.where(torch.isfinite(x), Vs, scale_v)
        Vs = torch.where(degenerate, scale_v, Vs)
        # pi_signal <= 0: max(c(1, tau_det2, r_z^2))
        Vs = torch.where(pi0, torch.stack([one, tmax, rz2max], 1).max(1).values, Vs)
        return Vs

    def _gate(self, r, eta2, rows, skip_mask=None):
        """compute_mixture_gate for the given rows (index tensor; None = all rows, with skip_mask = rows that need
        it); returns w (len(rows),P)."""
        if rows is None:
            rows = slice(None)
        self.gate_need = skip_mask
        nm1 = self.nm1[rows]
        rz = r / torch.sqrt(nm1)[:, None]
        lam = self.lam[rows]
        tau0 = torch.clamp(self.sigma2[rows][:, None] + (1.0 / self.Bref[rows] + lam)[:, None] * eta2, min=DBL_EPS)
        self.valid_sub = self.valid[rows]
        self.p_sub = self.p_real[rows]
        Vs = self._vsig(rz, tau0, self.pi_signal[rows])
        llr = _dnorm_log(rz, torch.sqrt(tau0)) - _dnorm_log(rz, torch.sqrt(Vs)[:, None])
        w = _plogis(self.gate_lpo[rows][:, None] + llr)
        return torch.where(torch.isfinite(w), w, torch.ones_like(w))

    def _lambda(self, r_fit_z, s_full, rows=None):
        """estimate_lambda_bias(r, s, sigma2, B, 'eb_mix', 'mle') per row."""
        sigma2, Bref = self.sigma2, self.Bref
        keep = self.valid & torch.isfinite(r_fit_z) & torch.isfinite(s_full) & (s_full > DBL_EPS)
        anyk = keep.any(1) & torch.isfinite(sigma2) & (sigma2 > DBL_EPS)
        r2 = torch.where(keep, r_fit_z * r_fit_z, torch.zeros_like(r_fit_z))
        s = torch.where(keep, s_full, torch.zeros_like(s_full))
        base = torch.where(keep, sigma2[:, None] + s / Bref[:, None], torch.ones_like(s))
        pos = (r2 - base) / torch.where(keep, s, torch.ones_like(s))
        posok = keep & torch.isfinite(pos) & (pos > 0)
        posmax = torch.where(posok, 10.0 * pos, torch.full_like(pos, -math.inf)).max(1).values
        fisher0 = 0.5 * torch.where(keep, (s / base) ** 2, torch.zeros_like(s)).sum(1)
        se_b = torch.where(fisher0 > 0, torch.sqrt(1.0 / fisher0), torch.full_like(fisher0, math.inf))
        one = torch.ones_like(fisher0)
        upper = torch.stack([one, 100.0 / Bref, posmax], 1).max(1).values

        def nll(lam):
            tau = base + lam[:, None] * s
            t = torch.log(tau) + r2 / tau
            return 0.5 * torch.where(keep, t, torch.zeros_like(t)).sum(1)

        for _ in range(400):                       # while (nll(upper) < nll(upper/2)) upper <- 10 upper
            grow = anyk & torch.isfinite(upper) & (upper < DBL_MAX / 10) & (nll(upper) < nll(upper / 2))
            if not bool(grow.any()):
                break
            upper = torch.where(grow, upper * 10.0, upper)
        if self.use_k:
            act = self.active if rows is None else self.active[rows]
            x = K.brent_lambda(r2, s, base, keep, upper,
                               torch.where(act, self.p_real, torch.zeros_like(self.p_real)).to(torch.int32))
        else:
            x = brent_fmin(nll, torch.zeros_like(upper), upper, upper.shape[0], upper.device, tol=1e-8)
        obj = nll(x)
        lam = torch.where(nll(torch.zeros_like(x)) <= obj, torch.zeros_like(x), x)
        lam = torch.where(lam < 0.1 * se_b, torch.zeros_like(lam), lam)
        return torch.where(anyk, lam, torch.zeros_like(lam))

    def update_lambda(self):
        """compute_R_mismatch_state: lambda from the full fitted residual (uses the running XtXr)."""
        b_full = (self.alpha * self.mu).sum(1)
        XtXr = self.XtXr
        sq = torch.sqrt(self.nm1)[:, None]
        r_fit_z = (self.Xty - XtXr) / sq
        v_g = torch.clamp(torch.where(self.valid, b_full * XtXr, torch.zeros_like(XtXr)).sum(1), min=0.0)
        s_full = XtXr * XtXr / self.nm1[:, None] + v_g[:, None]
        return self._lambda(r_fit_z, s_full)

    # ---- single effect update ------------------------------------------------------------------------
    def ser_update(self, l: int, rows_mask: torch.Tensor):
        cfg = self.cfg
        a_l, m_l = self.alpha[:, l], self.mu[:, l]
        XtXr_wo = self.XtXr - self.XtXb[:, l]
        r = self.Xty - XtXr_wo
        b_minus = (self.alpha * self.mu).sum(1) - a_l * m_l
        v_g = torch.clamp(torch.where(self.valid, b_minus * XtXr_wo, torch.zeros_like(r)).sum(1), min=0.0)
        eta2 = XtXr_wo * XtXr_wo / self.nm1[:, None]
        s = eta2 + v_g[:, None]
        invB = (1.0 / self.Bref)[:, None]
        lam = self.lam[:, None]
        infl = 1.0 + (invB + lam) * s / self.sigma2[:, None]
        gmask = rows_mask & (self.lam > 0)
        if self.use_k:
            # all rows in one launch; rows without the gate (lambda == 0 or inactive) are skipped inside the kernel
            w = self._gate(r, eta2, None, gmask)
            infl_g = 1.0 + (invB + w * lam) * s / self.sigma2[:, None]
            infl = torch.where(gmask[:, None], infl_g, infl)
        else:
            gate_rows = torch.nonzero(gmask).flatten()
            if gate_rows.numel():
                w = self._gate(r[gate_rows], eta2[gate_rows], gate_rows)
                infl_g = 1.0 + (invB[gate_rows] + w * lam[gate_rows]) * s[gate_rows] / self.sigma2[gate_rows][:, None]
                infl = infl.index_copy(0, gate_rows, infl_g)
        infl = torch.where(self.valid, infl, torch.ones_like(infl))
        betahat = r / self.d
        shat2 = self.sigma2[:, None] / self.d * infl
        V = self._optim_V(betahat, shat2, self.V[:, l], rows_mask)
        lbf = _ser_lbf(betahat, shat2, V[:, None])
        lbf = torch.where(self.valid, lbf, torch.zeros_like(lbf))
        lbf_model, w_, sw = _logsum2(lbf + self.lpw)
        alpha = w_ / sw[:, None]
        post_var = V[:, None] * shat2 / (V[:, None] + shat2)
        post_mean = post_var / shat2 * betahat
        post_mean2 = post_var + post_mean * post_mean
        XtXb = self.XtX_mv(alpha * post_mean)
        a3 = rows_mask[:, None]
        self.alpha[:, l] = torch.where(a3, alpha, self.alpha[:, l])
        self.mu[:, l] = torch.where(a3, post_mean, self.mu[:, l])
        self.mu2[:, l] = torch.where(a3, post_mean2, self.mu2[:, l])
        self.lbf_variable[:, l] = torch.where(a3, lbf, self.lbf_variable[:, l])
        self.V[:, l] = torch.where(rows_mask, V, self.V[:, l])
        self.lbf[:, l] = torch.where(rows_mask, lbf_model, self.lbf[:, l])
        self.XtXb[:, l] = torch.where(a3, XtXb, self.XtXb[:, l])
        self.XtXr = torch.where(a3, XtXr_wo + XtXb, self.XtXr)

    def ld_coherence(self):
        """compute_ser_ld_coherence(alpha_1): alpha' (Rs o Rs) alpha over keep = alpha >= 2 pi (renormalised),
        Rs = XtX/(n-1) = R/(csd csd')."""
        a = self.alpha[:, 0]
        keep = self.valid & (a > 0) & (a >= 2.0 * self.pi[:, None])
        anyk = keep.any(1)
        ak = torch.where(keep, a, torch.zeros_like(a))
        ak = ak / torch.where(anyk, ak.sum(1), torch.ones_like(anyk, dtype=a.dtype))[:, None]
        u = ak / (self.csd * self.csd)
        # only variants with alpha >= 2 pi enter (usually few): per row, form the keep x keep block of R
        # (as susieR does: data$XtX[keep, keep]) instead of the full p x p Hadamard square
        val = torch.zeros(u.shape[0], dtype=torch.float64, device=u.device)
        keep_cpu = keep.cpu()
        for r in torch.nonzero(anyk.cpu()).flatten().tolist():
            idx = torch.nonzero(keep_cpu[r]).flatten().to(u.device)
            b = r // self.T
            if self.ld.kind == "dense":
                G = self.ld.mat[b][idx][:, idx].to(torch.float64)
            else:
                Xk = self.ld.mat[b][:, idx].to(torch.float64)
                G = (Xk.T @ Xk) / self.ld.n_ref[b]
            uk = u[r, idx]
            val[r] = uk @ ((G * G) @ uk)
        val = torch.clamp(val, 0.0, 1.0)
        return torch.where(anyk, val, torch.ones_like(val))

    def pip_of(self, alpha):
        keep = (self.V > self.cfg.prior_tol)[:, :, None]
        return 1.0 - torch.where(keep, 1.0 - alpha, torch.ones_like(alpha)).prod(1)

    def select(self, idx):
        for name in ["valid", "p_real", "Bref", "nm1", "z_marg", "csd", "d", "Xty", "sigma2", "pi", "lpw", "alpha",
                     "mu", "mu2", "V", "lbf", "lbf_variable", "XtXr", "XtXb", "lam", "lam_diff", "zm", "gate_lpo",
                     "pi_signal", "niter", "converged", "failed", "active", "orig", "coherence", "lam_init"]:
            setattr(self, name, getattr(self, name)[idx])
        self.hist_alpha = [h[idx] for h in self.hist_alpha]
        self.hist_pip = [h[idx] for h in self.hist_pip]
        if self.prev_alpha is not None:
            self.prev_alpha = self.prev_alpha[idx]
        loc = idx[:: self.T] // self.T
        self.ld = self.ld.select(loc)


# ---------------------------------------------------------------------------------------------- driver
def _fit_bucket(zs, ns, lds, kind, n_refs, P, T, cfg: MMConfig, device, ld_dtype):
    """zs: list over loci of list over traits of z arrays; ns: same of n; lds: per locus LD (dense R or X)."""
    Bn = len(zs)
    NR = Bn * T
    f64 = torch.float64
    z = torch.zeros((NR, P), dtype=f64, device=device)
    valid = torch.zeros((NR, P), dtype=torch.bool, device=device)
    p_real = torch.zeros(NR, dtype=torch.int64, device=device)
    nn = torch.zeros(NR, dtype=f64, device=device)
    Bref = torch.zeros(NR, dtype=f64, device=device)
    for b in range(Bn):
        for t in range(T):
            r = b * T + t
            zi = np.nan_to_num(np.asarray(zs[b][t], dtype=np.float64), nan=0.0)
            p = len(zi)
            z[r, :p] = torch.as_tensor(zi, device=device)
            valid[r, :p] = True
            p_real[r] = p
            nn[r] = float(ns[b][t])
            Bref[r] = float(n_refs[b])
    ld = build_batch_ld(lds, kind, P, ld_dtype, device, symmetrize=(kind == "dense" and ld_dtype == torch.float64))
    if kind == "factor":
        ld.n_ref = torch.as_tensor(np.asarray([float(x) for x in n_refs]), dtype=f64, device=device)
    st = MMBucket(z, valid, p_real, nn, Bref, ld, T, cfg)
    results = [None] * NR
    p_np = p_real.cpu().numpy()

    def flush(rows):
        if not rows:
            return
        ri = torch.as_tensor(rows, device=device)
        # trim_null_effects
        V = st.V[ri]
        nul = V < cfg.prior_tol
        A = torch.where(nul[:, :, None], st.pi[ri][:, None, None].expand(-1, cfg.L, P), st.alpha[ri])
        A = torch.where(st.valid[ri][:, None, :], A, torch.zeros_like(A))
        LBF = torch.where(nul[:, :, None], torch.zeros_like(st.lbf_variable[ri]), st.lbf_variable[ri])
        V = torch.where(nul, torch.zeros_like(V), V)
        A, V, LBF = A.cpu().numpy(), V.cpu().numpy(), LBF.cpu().numpy()
        lbf_m = torch.where(nul, torch.zeros_like(st.lbf[ri]), st.lbf[ri]).cpu().numpy()
        MU = torch.where(nul[:, :, None], torch.zeros_like(st.mu[ri]), st.mu[ri]).cpu().numpy()
        nit, conv, fail = st.niter[ri].cpu().numpy(), st.converged[ri].cpu().numpy(), st.failed[ri].cpu().numpy()
        lam, coh, lam0 = st.lam[ri].cpu().numpy(), st.coherence[ri].cpu().numpy(), st.lam_init[ri].cpu().numpy()
        orig = st.orig[ri].cpu().numpy()
        for k, r in enumerate(rows):
            o = int(orig[k])
            p = int(p_np[o])
            alpha = A[k, :, :p]
            sets = _credible_sets(alpha, V[k], st.ld, r // T, p, cfg) if not fail[k] else \
                {"cs": [], "cs_index": [], "purity": [], "coverage": []}
            keep = V[k] > cfg.prior_tol
            pip = 1.0 - np.prod(1.0 - alpha[keep], axis=0) if keep.any() else np.zeros(p)
            results[o] = {"pip": pip, "alpha": alpha, "lbf_variable": LBF[k, :, :p], "V": V[k], "lbf": lbf_m[k],
                          "mu": MU[k, :, :p], "niter": int(nit[k]), "converged": bool(conv[k]),
                          "status": "failed_prior_check" if fail[k] else "ok", "sets": sets,
                          "lambda_bias": float(lam[k]), "lambda_init": float(lam0[k]), "ld_coherence": float(coh[k])}

    # ---- initialize_R_mismatch -------------------------------------------------------------------------
    allrows = st.active.clone()
    st.ser_update(0, allrows)
    lam0 = st.update_lambda()
    st.coherence = st.ld_coherence()
    st.lam_init = lam0 * st.coherence
    st.lam = st.lam_init.clone()
    st.prev_alpha = st.alpha.clone()
    # ---- IBSS ----------------------------------------------------------------------------------------
    for it in range(1, cfg.max_iter + 1):
        act = st.active
        for l in range(cfg.L):
            st.ser_update(l, act)
        old = st.lam
        new = st.update_lambda()
        st.lam = torch.where(act, new, old)
        st.lam_diff = torch.where(act, (new - old).abs(), st.lam_diff)
        st.niter = torch.where(act, torch.full_like(st.niter, it), st.niter)
        if cfg.check_prior:
            bad = act & (st.V > 100.0 * (st.zm[:, None] ** 2)).any(1)
            st.failed = st.failed | bad
            act = act & ~bad
        conv = torch.zeros_like(act)
        if it >= 2:
            cur = st.alpha
            cur_pip = st.pip_of(cur)
            if not st.hist_alpha:          # first check: history = [prev_alpha] with its PIP under the current V
                st.hist_alpha = [st.prev_alpha]
                st.hist_pip = [st.pip_of(st.prev_alpha)]
            h = len(st.hist_alpha)

            def sdiff(i):
                da = (cur - st.hist_alpha[i]).abs().amax((1, 2))
                dp = (cur_pip - st.hist_pip[i]).abs().amax(1)
                return torch.maximum(da, dp)
            conv = sdiff(h - 1) < cfg.tol
            cyc_done = conv.clone()
            new_alpha = cur
            averaged = torch.zeros_like(conv)
            for lag in range(2, min(cfg.cycle_window, h) + 1):
                idx = h - lag
                hit = (~cyc_done) & (sdiff(idx) < cfg.tol)
                if bool(hit.any()):
                    cyc = st.hist_alpha[h - lag + 1:] + [cur]
                    avg = sum(cyc[1:], cyc[0].clone()) / lag
                    new_alpha = torch.where(hit[:, None, None], avg, new_alpha)
                    averaged = averaged | hit
                cyc_done = cyc_done | hit
            conv = cyc_done & act
            averaged = averaged & act
            if bool(averaged.any()):
                st.alpha = torch.where(averaged[:, None, None], new_alpha, st.alpha)
                cur_pip = torch.where(averaged[:, None], st.pip_of(st.alpha), cur_pip)
                # R recomputes XtX (alpha_l mu_l) from the averaged alpha in the next sweep; XtXr is not refreshed
                ar = torch.nonzero(averaged).flatten()
                st.XtXb[ar] = _sub_mv(st, ar, st.alpha[ar] * st.mu[ar])
            conv = conv & ~(st.lam_diff > cfg.tol)
            st.hist_alpha.append(st.alpha.clone())
            st.hist_pip.append(cur_pip)
            if len(st.hist_alpha) > cfg.cycle_window:
                st.hist_alpha = st.hist_alpha[-cfg.cycle_window:]
                st.hist_pip = st.hist_pip[-cfg.cycle_window:]
        st.prev_alpha = st.alpha.clone()
        st.converged = st.converged | conv
        st.active = act & ~conv
        n_act = int(st.active.sum())
        if n_act == 0:
            break
        if n_act < cfg.compact_fraction * st.active.numel():
            # compaction keeps whole loci (all T traits) together
            loc_act = st.active.reshape(-1, T).any(1)
            keep_rows = torch.nonzero(loc_act.repeat_interleave(T)).flatten()
            done = torch.nonzero(~loc_act.repeat_interleave(T)).flatten().tolist()
            flush(done)
            st.select(keep_rows)
    flush([r for r in range(st.active.numel()) if results[int(st.orig[r])] is None])
    return results


def _sub_mv(st: MMBucket, rows: torch.Tensor, v: torch.Tensor):
    """XtX v for a subset of rows (v: (len(rows),K,P)); computed through the full-batch operator."""
    full = torch.zeros((st.alpha.shape[0],) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
    full[rows] = v
    return st.XtX_mv(full)[rows]


def susie_mm_batch(loci: list[dict], cfg: MMConfig | None = None, device=None, verbose: bool = False) -> list[list[dict]]:
    """Fit susieR 0.16.6 eb_mix SuSiE-RSS to every trait of every locus.

    loci: list of dicts with keys z (list of T arrays), n (list of T scalars), n_ref (int), and either
          R (p,p) dense or X (n_ref,p) factor (R = X'X/n_ref). Returns per locus a list of T result dicts.
    """
    cfg = cfg or MMConfig()
    check_ld_dtype(cfg.ld_dtype)
    device = device or default_device()
    if torch.device(device).type == "cuda":
        assert_no_tf32()
    N = len(loci)
    T = len(loci[0]["z"])
    bytes_ld = 4 if cfg.ld_dtype == torch.float32 else 8
    kinds = []
    for lc in loci:
        p = len(lc["z"][0])
        if lc.get("X") is not None and lc["n_ref"] < cfg.factor_ratio * p:
            kinds.append("factor")
        else:
            kinds.append("dense")
    out = [None] * N
    jobs = []                                    # (kind, bucket index list)
    for kind in ("factor", "dense"):
        idxs = sorted([i for i in range(N) if kinds[i] == kind], key=lambda i: len(loci[i]["z"][0]))
        bucket, p_min = [], None
        for i in idxs:
            lc = loci[i]
            p = len(lc["z"][0])
            per = (lc["n_ref"] * p if kind == "factor" else p * p) * bytes_ld + T * 18 * cfg.L * p * 8 * 2
            if bucket and (p > cfg.bucket_ratio * p_min or (len(bucket) + 1) * per > cfg.mem_budget_bytes
                           or len(bucket) >= cfg.max_batch):
                jobs.append((kind, bucket))
                bucket = []
            if not bucket:
                p_min = p
            bucket.append(i)
        if bucket:
            jobs.append((kind, bucket))

    def run(kind, bucket):
        P = max(len(loci[i]["z"][0]) for i in bucket)
        t0 = time.time()
        lds = []
        for i in bucket:
            lc = loci[i]
            if kind == "factor":
                lds.append(lc["X"])
            elif lc.get("R") is not None:
                lds.append(lc["R"])
            else:
                X = torch.as_tensor(np.asarray(lc["X"]), device=device).to(cfg.ld_dtype)
                lds.append((X.T @ X) / float(lc["n_ref"]))
        res = _fit_bucket([loci[i]["z"] for i in bucket], [loci[i]["n"] for i in bucket], lds, kind,
                          [loci[i]["n_ref"] for i in bucket], P, T, cfg, device, cfg.ld_dtype)
        if verbose:
            print(f"bucket kind={kind} B={len(bucket)} P={P} {time.time() - t0:.2f}s", flush=True)
        for k, i in enumerate(bucket):
            out[i] = res[k * T:(k + 1) * T]

    dev = torch.device(device)
    if cfg.streams > 1 and dev.type == "cuda" and len(jobs) > 1:
        # concurrent buckets: each worker thread owns a CUDA stream; a bucket's non-converged tail (few active
        # rows, launch-latency bound) then overlaps with other buckets' work. Results are stream-independent.
        import threading
        from concurrent.futures import ThreadPoolExecutor
        # largest buckets first (longest work) for better overlap
        jobs.sort(key=lambda j: -max(len(loci[i]["z"][0]) for i in j[1]) * len(j[1]))
        local = threading.local()

        def work(job):
            if not hasattr(local, "stream"):
                local.stream = torch.cuda.Stream(device=dev)
            with torch.cuda.stream(local.stream):
                run(*job)
            local.stream.synchronize()
        with ThreadPoolExecutor(max_workers=cfg.streams) as ex:
            list(ex.map(work, jobs))
    else:
        for job in jobs:
            run(*job)
    return out

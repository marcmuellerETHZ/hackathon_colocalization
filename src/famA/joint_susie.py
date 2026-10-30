"""JointSuSiE (Family A, DESIGN §5 A): batched GPU bivariate SuSiE-RSS with sharing states.

Model (z-scale, trait-major stacking z = [z1; z2]):
    z ~ N((I2 ⊗ R̃) b, V ⊗ R̃),   R̃ = (1 - η) R + η I,   V = [[1, ρ], [ρ, 1]] (ρ = rho_hat),
    b = Σ_l b_l, b_l = β_l ⊗ e_{j_l},  j_l ~ Uniform(1..p),  s_l ~ π over {0: null, 1: 1-only, 2: 2-only, 3: both},
    β_l | s ~ N(0, U_s):  U_0 = 0, U_1 = diag(σ²_l1, 0), U_2 = diag(0, σ²_l2), U_3 = [[σ²_l1, c], [c, σ²_l2]],
    c = ρ_u σ_l1 σ_l2 (ρ_u = 0 by default).
With diag(R) = 1 (standardised), the single-effect regression for effect l at variable j is exactly a bivariate
normal-means problem: bhat_j = r_j (residual z of both traits at j), bhat_j ~ N(β, V) -- no R^{-1} needed.

Variational IBSS as in susieR/mvsusieR; per effect l:
  * prior scales (σ²_l1, σ²_l2): empirical Bayes by coordinate-wise Brent (exact port of R's optimize, log scale,
    bounds [-30, 15]) with susieR's keep-old and null-check rules per coordinate ('optim'), or fixed ('fixed');
  * state prior π per locus: fixed ('fixed') or empirical Bayes ('eb', M3.5): after each sweep
    π_s ∝ Σ_l q_l(s) + κ π_ref,s (EM / MAP step under a Dirichlet(1 + κ π_ref) prior, κ = pi_prior_strength);
  * η (LD-error mixing weight) estimated jointly inside IBSS on the residual: after each sweep, η maximises
    the ELBO given q(b) (closed form in the eigenbasis of R; batched eigh / Gram eigh for genotype factors;
    fused Triton Brent, famA/eta_kernel.py) ('estimate_eta');
  * layer E (M3.5, ablation option 'ref_var'): per-variable variance inflation d_tj = 1 + δ_tj of the SER
    likelihood, δ_tj = Σ_i E[b_ti²] Var(r̂_ij), with Var(r̂) from an empirical table (famA/ref_ld_var_v2.json,
    pool resampling, v2 table) / n_ref; δ is updated after each sweep from the current posterior second moments.
Convergence: ELBO change (rule 'susie014': Δ < tol; 'susie016': iter > 1 and 0 <= Δ < tol, as mvsusieR) or
'alpha' (max |Δα| < tol_alpha; required with layer E, whose SER updates do not maximise the ELBO).

Locus outputs (DESIGN v4 §1 and §5 A, inclusion–exclusion), per effect l and variable j, with the marginal
noncentrality approximated as μ_t = β_t + m_{-l,t,j}, m_{-l} = R̃ E[b_{-l}] = z̃ - r_l (plug-in of the other effects'
posterior means; r_l = residual of effect l's last update):
  powered_t  = {|β_t| >= τ_t} ∩ {|μ_t| >= τ_t}          (own effect AND marginal, §1)
  nonweak_t  = {|β_t| >= w_t} ∩ {|μ_t| >= w_t}          (power >= 0.2; default: own-effect guard, P3-5(5);
                                                        nonweak_own = False gives the μ-only version)
  a_l = P(both, powered_1, powered_2), w_l = P(both, nonweak_1, nonweak_2) (blocking),
  p̃_l^1 = P(1-only, powered_1) + P(both, powered_1, |μ_2| < w_2), p̃_l^2 likewise, p̃_l^{1∨2} = p̃_l^1 + p̃_l^2;
  P_S = 1 - Π(1 - a), P_D = Π(1-w) - Π(1-w-p̃1) - Π(1-w-p̃2) + Π(1-w-p̃12), P_W = 1 - P_S - P_D,
  s = P_S/(P_S + P_D). Thresholds by trait TYPE (DESIGN v4 §1): GWAS-like τ = 6.29, w = 4.61; molecular τ = 5.26,
  w = 3.58. power_marginal = False gives the M3 own-effect-only events.
  Exclusion E_2 = Π_l (1 - Σ_{j in CS_1} α_lj P(l has |β2| >= w2 at j)) (own effect, weak threshold w of trait 2's
  type, DESIGN §5 layer D), CS_1 = union of the 95% variable credible sets of the effects with
  P(powered in trait 1) >= 0.5; E_1 symmetric (NaN if CS_t is empty).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from famG.brent import brent_fmin
from famG.ld import BatchLD, build_batch_ld
from famG.precision import assert_no_tf32, check_ld_dtype, default_device
from famG.susie_rss import _credible_sets, SusieRSSConfig

from .bvn import bvn_set_prob, marginal, nonweak_spec, powered_spec, set_prob, weak_spec

try:
    from .ser_kernel import eb_coordinate
except Exception:  # pragma: no cover (no triton)
    eb_coordinate = None
try:
    from .eta_kernel import eta_update as eta_update_triton
except Exception:  # pragma: no cover (no triton)
    eta_update_triton = None

LOG2PI = math.log(2 * math.pi)
# DESIGN v4 §1 (frozen): (τ = 0.8-power, w = 0.2-power) thresholds by trait TYPE; equal to sim.params TAU/WEAK
# (asserted in tests/famA/test_power_events.py).
TYPE_THRESHOLDS = {"gwas": (6.29, 4.61), "mol": (5.26, 3.58)}
REF_VAR_TABLE = Path(__file__).resolve().parent / "ref_ld_var_v2.json"   # v2: monomorphic draws excluded (P3)


@dataclass
class JointConfig:
    L: int = 10
    pi: tuple = (0.25, 0.25, 0.25, 0.25)        # state prior (null, 1-only, 2-only, both): fixed value / EB reference
    pi_mode: str = "eb"                          # 'eb' (per-locus EM/MAP, M3.5) | 'fixed'
    pi_prior_strength: float = 4.0               # κ of the Dirichlet(1 + κ π_ref) prior in 'eb' mode
    prior: str = "optim"                         # 'optim' (EB per effect & trait) | 'fixed'
    sigma2_init: tuple = (50.0, 50.0)            # z-scale prior variances (initial / fixed values)
    U_fixed: tuple | None = None                 # optional ((u11,u12,u22) for states 1,2,3) for 'fixed'
    rho_u: float = 0.0                           # effect correlation in the shared state
    estimate_eta: bool = False
    eta_init: float = 0.0
    eta_bounds: tuple = (1e-8, 0.95)
    eig_dtype: torch.dtype = torch.float64       # dense-LD eigen-decomposition for η (float32 = fast option)
    eig_chunk: int = 4                           # loci per batched eigh call (bounds GPU memory)
    eig_rel_tol: float = 1e-10                   # eigenvalues <= tol * max are null-space directions (float64)
    eig_rel_tol_fp32: float = 1e-6               # same for the float32 eigen-decomposition
    eta_min_fp32: float = 1e-3                   # η floor with the float32 eigen-decomposition (spectrum noise)
    eta_floor: float = 1e-3                      # P3-5(3): η >= eta_floor on ALL rungs, also when η is not estimated
                                                 # (fixed η = max(eta_init, eta_floor)); 0 = M3.5 behaviour (identity)
    multistart: bool = True                      # P3-5(1): default start + 'no-shared-first' start, keep higher ELBO
    noshare_sweeps: int = 5                      # sweeps with the shared state disabled in the second start
    per_trait_ld: bool = True                    # P3-5(2): per-trait mean map R̃_t when a locus gives a 2nd LD
    ref_var: bool = False                        # layer E: empirical reference-size variance term (ablation)
    ref_var_table: str = str(REF_VAR_TABLE)
    ref_var_topk: int = 32                       # variables (per trait) carrying E[b²] used for δ
    z_adjust: bool = True                        # PVE adjustment z * sqrt((n-1)/(z^2+n-2)), per trait
    rho_clip: float = 0.95
    check_null_threshold: float = 0.0
    prior_tol: float = 1e-9
    max_iter: int = 100
    tol: float = 1e-3
    tol_alpha: float = 1e-4
    tol_rel: float = 1e-6                        # 'rel': |ΔELBO| < tol_rel·|ELBO| also counts as converged
    convergence: str = "rel"                     # 'rel' (P3-5(4): Δ < tol or |Δ| < tol_rel|ELBO| or max|Δα| < tol_alpha)
                                                 # | 'susie014' | 'susie016' | 'alpha'
    default_types: tuple = ("gwas", "mol")       # trait types when a locus has none (INTERFACES trait{1,2}_type)
    tau: tuple | None = None                     # optional override of the type thresholds (all loci)
    weak: tuple | None = None
    power_marginal: bool = True                  # power events use own effect AND marginal μ (DESIGN v4 §1)
    nonweak_own: bool = True                     # 'power >= 0.2' (blocking) also requires own |β| >= w (README)
    bvn_gather_tol: float = 1e-13                # bivariate events skipped where α·P(both|j) <= tol
    coverage: float = 0.95
    min_abs_corr: float = 0.5
    powered_cs_min: float = 0.5                  # effects entering CS_t for the exclusion probabilities
    ld_dtype: torch.dtype = torch.float32
    mem_budget_bytes: float = 6e9
    bucket_ratio: float = 1.10
    max_batch: int = 4096
    compact_fraction: float = 0.5
    refill_fraction: float = 0.25                # continuous batching: refill when >= this share of rows is done
    return_details: bool = True
    use_triton: bool = True                      # fused EB and η kernels on CUDA (famA/ser_kernel.py, eta_kernel.py)
    eb_fast_exp: bool = False                    # float32 exp inside the EB kernel (speed option, off)


# ------------------------------------------------------------------------------------------------------
# single-effect regression primitives (all float64, shapes: r (B,2,P), V entries (B,), U entries (B,3 states,3))
# Optional per-variable variance inflation d (B,2,P) (layer E): the SER at j uses r_j ~ N(β, D_j V D_j),
# D_j = diag(√d_1j, √d_2j); equivalently γ = D_j^{-1} β with r̃_j = D_j^{-1} r_j ~ N(γ, V), γ ~ N(0, D_j^{-1} U D_j^{-1}).
# ------------------------------------------------------------------------------------------------------
def _U_from_scales(s2, rho_u):
    """s2 (B,2) -> U entries (B,3,3): state k in {1,2,3}, entries (u11,u12,u22)."""
    s1, s2b = s2[:, 0], s2[:, 1]
    z = torch.zeros_like(s1)
    c = rho_u * torch.sqrt(s1 * s2b)
    U1 = torch.stack([s1, z, z], -1)
    U2 = torch.stack([z, z, s2b], -1)
    U3 = torch.stack([s1, c, s2b], -1)
    return torch.stack([U1, U2, U3], 1)


def _u_entries(U, d):
    """Prior entries in γ-space, shapes (B,3,1) (d None) or (B,3,P)."""
    u11, u12, u22 = U[..., 0][..., None], U[..., 1][..., None], U[..., 2][..., None]
    if d is None:
        return u11, u12, u22
    d1, d2 = d[:, None, 0], d[:, None, 1]
    return u11 / d1, u12 / torch.sqrt(d1 * d2), u22 / d2


def _state_lbf_g(rg, rho, u11, u12, u22):
    r1, r2 = rg[:, 0], rg[:, 1]                                 # (B,P)
    rho_ = rho[:, None, None]
    m11, m12, m22 = 1.0 + u11, rho_ + u12, 1.0 + u22
    det = m11 * m22 - m12 * m12
    detS = (1.0 - rho * rho)[:, None, None]
    a1, a2 = r1[:, None, :], r2[:, None, :]
    qM = (m22 * a1 * a1 - 2.0 * m12 * a1 * a2 + m11 * a2 * a2) / det
    qS = (a1 * a1 - 2.0 * rho_ * a1 * a2 + a2 * a2) / detS
    lbf = -0.5 * torch.log(det / detS) - 0.5 * (qM - qS)
    return lbf, (m11, m12, m22, det)


def _state_lbf(r, rho, U, d=None):
    """log BF of each non-null state at each variable: (B,3,P). Also returns M^{-1} entries."""
    rg = r if d is None else r / torch.sqrt(d)
    return _state_lbf_g(rg, rho, *_u_entries(U, d))


def _var_lbf(lbf_s, logpi):
    """Variable-level log BF (B,P) = log(π0 + Σ_s π_s BF_s), plus per-state log weights (B,4,P)."""
    B, _, P = lbf_s.shape
    l0 = torch.zeros((B, 1, P), dtype=lbf_s.dtype, device=lbf_s.device)
    allv = torch.cat([l0, lbf_s], 1) + logpi[:, :, None]       # (B,4,P)
    m = allv.max(1, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    lj = (m + torch.log(torch.exp(allv - m).sum(1, keepdim=True)))[:, 0]
    return lj, allv - lj[:, None, :]


def _loglik(r, rho, U, logpi, lpw, d=None):
    lbf_s, _ = _state_lbf(r, rho, U, d)
    lj, _ = _var_lbf(lbf_s, logpi)
    lpo = lj + lpw
    m = lpo.max(1, keepdim=True).values
    return (m + torch.log(torch.exp(lpo - m).sum(1, keepdim=True)))[:, 0]


def _posterior(r, rho, U, logpi, lpw, d=None):
    """Full SER posterior. Returns dict with alpha (B,P), lbf_model (B,), lbf_var (B,P), wst (B,4,P) state
    weights given j, mu (B,3,2,P) state means of β, Sig (B,3,3,Pk) state covariances of β (entries s11, s12, s22;
    Pk = 1 without inflation, P with), Eb (B,2,P) = E[β|j], tr (B,P) = tr(V^{-1} E[ββ'|j]), e2 (B,2,P) = E[β_t²|j],
    and the γ-space quantities Eb_g, tr_g, r_g used for the KL (equal to the β ones when d is None)."""
    rg = r if d is None else r / torch.sqrt(d)
    u11, u12, u22 = _u_entries(U, d)
    lbf_s, (m11, m12, m22, det) = _state_lbf_g(rg, rho, u11, u12, u22)
    lj, lw = _var_lbf(lbf_s, logpi)
    lpo = lj + lpw
    mx = lpo.max(1, keepdim=True).values
    ex = torch.exp(lpo - mx)
    sw = ex.sum(1, keepdim=True)
    alpha = ex / sw
    lbf_model = (mx + torch.log(sw))[:, 0]
    wst = torch.exp(lw)                                        # (B,4,P)
    i11, i12, i22 = m22 / det, -m12 / det, m11 / det
    k11 = u11 * i11 + u12 * i12
    k12 = u11 * i12 + u12 * i22
    k21 = u12 * i11 + u22 * i12
    k22 = u12 * i12 + u22 * i22
    r1, r2 = rg[:, None, 0], rg[:, None, 1]
    mu1 = k11 * r1 + k12 * r2                                  # (B,3,P)
    mu2 = k21 * r1 + k22 * r2
    s11 = u11 - (k11 * u11 + k12 * u12)                        # (B,3,Pk)
    s12 = u12 - (k11 * u12 + k12 * u22)
    s22 = u22 - (k21 * u12 + k22 * u22)
    w = wst[:, 1:]                                             # (B,3,P)
    rho_ = rho[:, None]
    detS = 1.0 - rho_ * rho_

    def moments(m1, m2, a11, a12, a22):
        Eb1 = (w * m1).sum(1)
        Eb2 = (w * m2).sum(1)
        e11 = (w * (a11 + m1 * m1)).sum(1)
        e12 = (w * (a12 + m1 * m2)).sum(1)
        e22 = (w * (a22 + m2 * m2)).sum(1)
        return torch.stack([Eb1, Eb2], 1), (e11 - 2.0 * rho_ * e12 + e22) / detS, torch.stack([e11, e22], 1)

    Eb_g, tr_g, e2_g = moments(mu1, mu2, s11, s12, s22)
    if d is None:
        Eb, tr, e2 = Eb_g, tr_g, e2_g
        mb1, mb2, b11, b12, b22 = mu1, mu2, s11, s12, s22
    else:
        q1, q2 = torch.sqrt(d[:, None, 0]), torch.sqrt(d[:, None, 1])   # (B,1,P)
        mb1, mb2 = mu1 * q1, mu2 * q2
        b11, b12, b22 = s11 * q1 * q1, s12 * q1 * q2, s22 * q2 * q2
        Eb, tr, e2 = moments(mb1, mb2, b11, b12, b22)
    return dict(alpha=alpha, lbf_model=lbf_model, lbf_var=lj, wst=wst, mu=torch.stack([mb1, mb2], 2),
                Sig=torch.stack([b11, b12, b22], 2), Eb=Eb, tr=tr, e2=e2, Eb_g=Eb_g, tr_g=tr_g, r_g=rg)


def _vinv_dot(a, b, rho):
    """Σ_j a_jᵀ V^{-1} b_j for (B,2,P) arrays -> (B,)."""
    rho_ = rho[:, None]
    return ((a[:, 0] * b[:, 0] - rho_ * (a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]) + a[:, 1] * b[:, 1]).sum(1)
            / (1.0 - rho * rho))


def thresholds_for_types(types, cfg: JointConfig) -> np.ndarray:
    """(N,4) array [τ1, τ2, w1, w2] from per-locus (type1, type2); cfg.tau / cfg.weak override if given."""
    out = np.zeros((len(types), 4))
    for i, (t1, t2) in enumerate(types):
        for t, k in ((t1, 0), (t2, 1)):
            if t not in TYPE_THRESHOLDS:
                raise ValueError(f"unknown trait type {t!r} (expected one of {sorted(TYPE_THRESHOLDS)})")
            out[i, k], out[i, 2 + k] = TYPE_THRESHOLDS[t]
    if cfg.tau is not None:
        out[:, 0], out[:, 1] = cfg.tau
    if cfg.weak is not None:
        out[:, 2], out[:, 3] = cfg.weak
    return out


@lru_cache(maxsize=4)
def load_ref_var_table(path: str):
    """Empirical n·Var(r̂) table: (r2_edges, maf_edges, n_list, kappa_by_n (Nn, nb, nm), kappa_pooled_maf (Nn, nb)).
    Empty cells (no pairs or negative rounding) are replaced by the MAF-pooled value of the same r² bin and n."""
    res = json.loads(Path(path).read_text())
    n_list = [int(n) for n in res["n_list"]]
    K = np.array([np.array(res["kappa_by_n"][str(n)], dtype=float) for n in n_list])      # (Nn, nb, nm)
    C = np.array(res["counts"], dtype=float)                                                # (nb, nm)
    Kz = np.where(np.isfinite(K), K, 0.0)
    pooled = (Kz * C[None]).sum(2) / np.maximum(C.sum(1)[None], 1.0)                       # (Nn, nb)
    bad = ~np.isfinite(K) | (C[None] == 0)
    K = np.where(bad, pooled[:, :, None], K)
    K = np.clip(K, 0.0, None)
    pooled = np.clip(pooled, 0.0, None)
    return (np.array(res["r2_edges"], float), np.array(res["maf_edges"], float), np.array(n_list, float), K, pooled)


# ------------------------------------------------------------------------------------------------------
class _JointBucket:
    def __init__(self, z, valid, p_real, n, ld: BatchLD, rho, cfg: JointConfig, thr=None, nref_e=None, maf=None,
                 ld2: BatchLD | None = None, noshare=None):
        self.cfg = cfg
        dev = z.device
        f64 = torch.float64
        B, _, P = z.shape
        L = cfg.L
        self.ld, self.valid, self.p_real = ld, valid, p_real
        self.ld2 = ld2                               # per-trait LD (P3-5(2)): trait-2 mean map; None = one R

        def _csd(l_):
            c = torch.sqrt(l_.diag())
            return torch.where((c == 0) | ~valid, torch.ones_like(c), c)
        csd = _csd(ld)
        self.csd = csd
        self.csd2 = _csd(ld2) if ld2 is not None else csd
        if cfg.z_adjust and n is not None:
            adj = (n[:, :, None] - 1.0) / (z * z + n[:, :, None] - 2.0)
            z = torch.sqrt(adj) * z
        z = z / torch.stack([csd, self.csd2], 1)
        self.z = torch.where(valid[:, None, :], z, torch.zeros_like(z))
        self.rho = rho.clamp(-cfg.rho_clip, cfg.rho_clip)
        if thr is None:
            thr = torch.as_tensor(thresholds_for_types([cfg.default_types] * B, cfg), dtype=f64, device=dev)
        self.thr = thr
        pi = torch.as_tensor(cfg.pi, dtype=f64, device=dev)
        pi = pi / pi.sum()
        self.pi_ref = pi
        self.logpi = torch.log(pi)[None, :].expand(B, 4).clone()
        self.lpw = torch.where(valid, torch.log(1.0 / p_real.to(f64))[:, None].expand(B, P), torch.full((B, P), -math.inf, dtype=f64, device=dev))
        self.alpha = torch.where(valid[:, None, :], (1.0 / p_real.to(f64))[:, None, None].expand(B, L, P), torch.zeros(1, dtype=f64, device=dev)).clone()
        self.Eb = torch.zeros((B, L, 2, P), dtype=f64, device=dev)
        self.RB = torch.zeros((B, L, 2, P), dtype=f64, device=dev)   # R_std @ Eb_l (pure R, no eta)
        self.RBsum = torch.zeros((B, 2, P), dtype=f64, device=dev)
        self.rlast = torch.zeros((B, L, 2, P), dtype=f64, device=dev)  # residual at the last update of each effect
        self.smass = torch.zeros((B, L, 4), dtype=f64, device=dev)     # state posterior mass per effect
        self.quad = torch.zeros((B, L), dtype=f64, device=dev)
        self.KL = torch.full((B, L), math.nan, dtype=f64, device=dev)
        self.lbf = torch.full((B, L), math.nan, dtype=f64, device=dev)
        self.s2 = torch.as_tensor(cfg.sigma2_init, dtype=f64, device=dev)[None, None, :].expand(B, L, 2).clone()
        eta0 = max(float(cfg.eta_init), cfg.eta_bounds[0]) if cfg.estimate_eta else float(cfg.eta_init)
        eta0 = max(eta0, float(cfg.eta_floor))
        self.eta = torch.full((B,), eta0, dtype=f64, device=dev)
        self.elbo_prev = torch.full((B,), -math.inf, dtype=f64, device=dev)
        self.elbo_hist = [[] for _ in range(B)]
        self.eta_hist = [[] for _ in range(B)]
        self.niter = torch.zeros(B, dtype=torch.int64, device=dev)
        self.converged = torch.zeros(B, dtype=torch.bool, device=dev)
        self.active = torch.ones(B, dtype=torch.bool, device=dev)
        self.orig = torch.arange(B, device=dev)
        # multi-start (P3-5(1)): number of initial sweeps with the shared state disabled (0 = default start)
        self.noshare = (torch.zeros(B, dtype=torch.int64, device=dev) if noshare is None
                        else torch.as_tensor(noshare, dtype=torch.int64, device=dev))
        self.eig = None
        # layer E
        self.dvar = None
        if cfg.ref_var:
            if nref_e is None:
                raise ValueError("layer E (ref_var) needs the reference size n_ref of every locus")
            self.nref_e = nref_e.to(f64)
            self.maf = maf                                          # (B,P) reference MAF or None
            self.dvar = torch.ones((B, 2, P), dtype=f64, device=dev)
            self.dlast = torch.ones((B, L, 2, P), dtype=f64, device=dev)
            self.E2 = torch.zeros((B, L, 2, P), dtype=f64, device=dev)
            self._ref_tab = self._ref_var_tensors()
        if cfg.estimate_eta:
            self._eigen()

    _SEL = ["valid", "p_real", "csd", "csd2", "noshare", "z", "rho", "thr", "logpi", "lpw", "alpha", "Eb", "RB", "RBsum", "rlast", "smass",
            "quad", "KL", "lbf", "s2", "eta", "elbo_prev", "niter", "converged", "active", "orig"]

    # ---- LD products ----------------------------------------------------------------------------------
    def R_mv(self, v):
        """R_std v for v (B,2,P) (trait-major); with per-trait LD, trait t uses its own R_t."""
        if self.ld2 is None:
            return self.ld.matvec(v / self.csd[:, None, :]) / self.csd[:, None, :]
        o1 = self.ld.matvec(v[:, 0] / self.csd) / self.csd
        o2 = self.ld2.matvec(v[:, 1] / self.csd2) / self.csd2
        return torch.stack([o1, o2], 1)

    def Rt_from(self, RB, Eb):
        e = self.eta[:, None, None]
        return (1.0 - e) * RB + e * Eb

    # ---- eigen decomposition for the eta likelihood ---------------------------------------------------
    def _spectrum(self, ld, csd_v, zc_all):
        """(lam (B,K), zt (B,K,T), exact) of R_std for one LD and trait slice zc_all (B,T,P) (see _eigen)."""
        cfg = self.cfg
        B, P = self.z.shape[0], self.z.shape[2]
        dev, f64 = self.z.device, torch.float64
        csd = torch.where(self.valid, csd_v, torch.full_like(csd_v, math.inf))   # padded -> scaled to 0
        use_gram = ld.kind == "factor" and ld.mat.shape[1] <= P
        K = ld.mat.shape[1] if use_gram else P
        T = zc_all.shape[1]
        lam_all = torch.zeros((B, K), dtype=f64, device=dev)
        zt_all = torch.zeros((B, K, T), dtype=f64, device=dev)
        ch = max(1, int(cfg.eig_chunk))
        for c0 in range(0, B, ch):
            sl = slice(c0, min(B, c0 + ch))
            zc = zc_all[sl]                                                  # (b,T,P), padded entries 0
            if use_gram:
                Xs = ld.mat[sl].to(f64) / csd[sl][:, None, :] / torch.sqrt(ld.n_ref[sl])[:, None, None]  # (b,n,P)
                G = Xs @ Xs.transpose(1, 2)
                lam, U = torch.linalg.eigh(G)                                # (b,n), (b,n,n)
                lam = lam.clamp_min(0.0)
                Xz = Xs @ zc.transpose(1, 2)                                 # (b,n,T)
                proj = U.transpose(1, 2) @ Xz                                # (b,n,T) = Λ^{1/2} Qᵀz̃
                inv = torch.where(lam > 0, 1.0 / torch.sqrt(torch.where(lam > 0, lam, torch.ones_like(lam))),
                                  torch.zeros_like(lam))
                zt = proj * inv[..., None]
            else:
                if ld.kind == "dense":
                    Rm = ld.mat[sl].to(cfg.eig_dtype)
                    sc = (1.0 / csd[sl]).to(cfg.eig_dtype)
                    Rm = Rm * sc[:, :, None] * sc[:, None, :]
                else:
                    Xs = ld.mat[sl].to(cfg.eig_dtype) / csd[sl][:, None, :].to(cfg.eig_dtype)
                    Rm = (Xs.transpose(1, 2) @ Xs) / ld.n_ref[sl][:, None, None].to(cfg.eig_dtype)
                lam, Q = torch.linalg.eigh(Rm)
                lam = lam.to(f64).clamp_min(0.0)
                zt = (Q.transpose(1, 2) @ zc.transpose(1, 2).to(cfg.eig_dtype)).to(f64)   # (b,P,T)
                del Rm, Q
            lam_all[sl] = lam
            zt_all[sl] = zt
        exact = bool(use_gram or cfg.eig_dtype == torch.float64)
        return lam_all, zt_all, exact

    def _eigen(self):
        """Spectrum of R_std for the η likelihood, batched on the GPU (M3.5):
        eig_lam (B,K) >= 0, eig_q (B,K) = (Qᵀz̃)ᵀV^{-1}(Qᵀz̃) per component, eig_qnull (B,), eig_nnull (B,).

        * factor LD (R = X̃ᵀX̃, X̃ = X diag(1/csd)/√n_ref) with n_ref <= p: eigen-decomposition of the
          n_ref × n_ref Gram matrix G = X̃X̃ᵀ in float64 (exact; same non-zero spectrum as R); the projections are
          Qᵀz̃ = Λ^{-1/2} Uᵀ X̃ z̃, and the p − rank null-space directions carry eigenvalue η.
        * dense LD (or factor with n_ref > p, formed as X̃ᵀX̃): batched torch.linalg.eigh in cfg.eig_dtype
          (float64 = exact, default; float32 = fast option, see README), in chunks of cfg.eig_chunk loci.
        Padded variables (j >= p_real) are zero rows/columns of R and zero entries of z̃; their (zero) eigenvalues
        are removed from the null-space count explicitly.

        Per-trait LD (P3-5(2)): the spectra of R_1 and R_2 are concatenated; the η terms use log|R̃_1| + log|R̃_2|
        and the quadratic form [z̃_1ᵀR̃_1⁻¹z̃_1 + z̃_2ᵀR̃_2⁻¹z̃_2 − ρ z̃_1ᵀ(R̃_1⁻¹ + R̃_2⁻¹)z̃_2]/(1 − ρ²): exact for ρ = 0
        and for R_1 = R_2 (then equal to the one-LD terms); for ρ ≠ 0 and R_1 ≠ R_2 it is an approximation (the
        working noise covariance is not specified beyond its SER margins; README)."""
        cfg = self.cfg
        rho = self.rho
        r_ = rho[:, None]
        den = 1 - r_ * r_
        thr_of = lambda ex: cfg.eig_rel_tol if ex else cfg.eig_rel_tol_fp32
        if self.ld2 is None:
            lam_all, zt, exact = self._spectrum(self.ld, self.csd, self.z)
            q_all = (zt[..., 0] ** 2 - 2 * r_ * zt[..., 0] * zt[..., 1] + zt[..., 1] ** 2) / den
            lmax = lam_all.max(1, keepdim=True).values
            mask = lam_all > thr_of(exact) * lmax
            qtot = _vinv_dot(self.z, self.z, self.rho)
            qnull = (qtot - (q_all * mask).sum(1)).clamp_min(0.0)
            nnull = self.p_real.to(torch.float64) - mask.sum(1).to(torch.float64)
            self.eig_ldmult = 2.0
        else:
            lam_l, q_l, m_l, qn, nn = [], [], [], 0.0, 0.0
            exact = True
            cross = (self.z[:, 0] * self.z[:, 1]).sum(1)
            for ld_, csd_, t in ((self.ld, self.csd, 0), (self.ld2, self.csd2, 1)):
                lam, zt, ex = self._spectrum(ld_, csd_, self.z)          # both traits projected on R_t's basis
                exact = exact and ex
                q = (zt[..., t] ** 2 - r_ * zt[..., 0] * zt[..., 1]) / den
                mk = lam > thr_of(ex) * lam.max(1, keepdim=True).values
                qt = ((self.z[:, t] ** 2).sum(1) - rho * cross) / den[:, 0]
                qn = qn + (qt - (q * mk).sum(1))
                nn = nn + self.p_real.to(torch.float64) - mk.sum(1).to(torch.float64)
                lam_l.append(lam); q_l.append(q); m_l.append(mk)
            lam_all, q_all, mask = torch.cat(lam_l, 1), torch.cat(q_l, 1), torch.cat(m_l, 1)
            qnull, nnull = qn.clamp_min(0.0), nn
            self.eig_ldmult = 1.0
        self.eig_lam = lam_all
        self.eig_q = q_all
        self.eig_mask = mask
        self.eig_qnull = qnull
        self.eig_nnull = nnull
        self.eig = True
        self.eig_exact = exact
        if not self.eig_exact:     # float32 spectrum: eigenvalues below ~1e-6·λmax are noise, so η is floored
            self.eta = self.eta.clamp_min(cfg.eta_min_fp32)

    def _eta_terms(self, eta):
        """-½ log|V ⊗ R̃| - ½ zᵀ(V^{-1} ⊗ R̃^{-1})z - p log 2π (η-dependent ELBO constant) for (B,) eta."""
        e = eta[:, None]
        lt = (1.0 - e) * self.eig_lam + e
        lt = torch.where(self.eig_mask, lt, torch.ones_like(lt))
        logdetR = (torch.log(lt) * self.eig_mask).sum(1) + self.eig_nnull * torch.log(eta)
        quad = (self.eig_q / lt * self.eig_mask).sum(1) + self.eig_qnull / eta
        logdetV = self.p_real.to(torch.float64) * torch.log(1.0 - self.rho * self.rho)
        return -0.5 * (logdetV + self.eig_ldmult * logdetR) - 0.5 * quad - self.p_real.to(torch.float64) * LOG2PI

    # ---- layer E: reference-size variance ----------------------------------------------------------------
    def _ref_var_tensors(self):
        r2e, mafe, n_list, K, pooled = load_ref_var_table(self.cfg.ref_var_table)
        dev = self.z.device
        # table of the calibrated reference size closest (log scale) to each locus' n_ref
        ln = np.log(n_list)
        nidx = torch.as_tensor(np.abs(np.log(np.maximum(self.nref_e.cpu().numpy(), 1.0))[:, None] - ln[None]).argmin(1),
                               device=dev)
        return dict(r2_inner=torch.as_tensor(r2e[1:-1], dtype=torch.float64, device=dev),
                    maf_inner=torch.as_tensor(mafe[1:-1], dtype=torch.float64, device=dev),
                    K=torch.as_tensor(K, dtype=torch.float64, device=dev),
                    pooled=torch.as_tensor(pooled, dtype=torch.float64, device=dev), nidx=nidx)

    def _R_rows(self, idx):
        """Standardised R rows for variable indices idx (B,M) -> (B,M,P) float64 (reference LD as given)."""
        ld = self.ld
        if ld.kind == "dense":
            P = ld.mat.shape[-1]
            rows = torch.gather(ld.mat, 1, idx[:, :, None].expand(-1, -1, P)).to(torch.float64)
        else:
            Xi = torch.gather(ld.mat, 2, idx[:, None, :].expand(-1, ld.mat.shape[1], -1))       # (B,n,M)
            rows = (Xi.transpose(1, 2).to(torch.float64) @ ld.mat.to(torch.float64)) / ld.n_ref[:, None, None]
        ci = torch.gather(self.csd, 1, idx)
        return rows / ci[:, :, None] / self.csd[:, None, :]

    def update_delta(self):
        """δ_tk = Σ_{i in top-K of E[b_t²]} E[b_ti²] · κ(r²_ik, min MAF)/n_ref (κ = empirical n·Var(r̂), M3.5)."""
        cfg, tab = self.cfg, self._ref_tab
        B, _, P = self.z.shape
        E2 = self.E2.sum(1)                                          # (B,2,P) E[b_t²] at each variable
        K = min(cfg.ref_var_topk, P)
        top = E2.topk(K, dim=2).indices                              # (B,2,K)
        idx = top.reshape(B, 2 * K)
        rows = self._R_rows(idx).reshape(B, 2, K, P)
        r2 = (rows * rows).clamp(0.0, 1.0)
        bi = torch.bucketize(r2, tab["r2_inner"], right=True)        # (B,2,K,P)
        nsel = tab["nidx"][:, None, None, None].expand_as(bi)
        if self.maf is not None:
            maf = self.maf
            mi = torch.minimum(torch.gather(maf, 1, idx).reshape(B, 2, K)[..., None], maf[:, None, None, :])
            mb = torch.bucketize(mi, tab["maf_inner"], right=True)
            kap = tab["K"][nsel, bi, mb]
        else:
            kap = tab["pooled"][nsel, bi]
        v = kap / self.nref_e[:, None, None, None]
        self_pair = torch.arange(P, device=v.device)[None, None, None, :] == top[..., None]
        v = torch.where(self_pair, torch.zeros_like(v), v)
        w = torch.gather(E2, 2, top)                                 # (B,2,K)
        delta = (w[..., None] * v).sum(2)                            # (B,2,P)
        d = torch.where(self.valid[:, None, :], 1.0 + delta, torch.ones_like(delta))
        self.dvar = torch.where(self.active[:, None, None], d, self.dvar)

    # ---- SER with EB ----------------------------------------------------------------------------------
    def _U(self, s2):
        cfg = self.cfg
        if cfg.prior == "fixed" and cfg.U_fixed is not None:
            U = torch.as_tensor(cfg.U_fixed, dtype=torch.float64, device=s2.device)
            return U[None].expand(s2.shape[0], 3, 3)
        return _U_from_scales(s2, cfg.rho_u)

    def ser(self, r, s2_old):
        cfg = self.cfg
        B = r.shape[0]
        s2 = s2_old.clone()
        d = self.dvar
        if cfg.prior == "optim" and r.is_cuda and cfg.use_triton and eb_coordinate is not None and d is None:
            for t in (0, 1):
                new = eb_coordinate(r, self.lpw, s2, self.rho, self.logpi, self.active, t, cfg.rho_u,
                                    cfg.check_null_threshold, fast_exp=cfg.eb_fast_exp)
                s2 = s2.clone()
                s2[:, t] = new
        elif cfg.prior == "optim":
            for t in (0, 1):
                def ll_at(val, t=t, base=s2):
                    s = base.clone()
                    s[:, t] = val
                    return _loglik(r, self.rho, _U_from_scales(s, cfg.rho_u), self.logpi, self.lpw, d)
                f = lambda lx: -ll_at(torch.exp(lx))
                lx = brent_fmin(f, -30.0, 15.0, B, r.device)
                old = s2[:, t]
                keep = f(lx) > f(torch.log(old))
                lx = torch.where(keep, torch.log(old), lx)
                new = torch.exp(lx)
                l0 = ll_at(torch.zeros_like(new))
                l1 = ll_at(new)
                new = torch.where(l0 + cfg.check_null_threshold >= l1, torch.zeros_like(new), new)
                s2 = s2.clone()
                s2[:, t] = new
        post = _posterior(r, self.rho, self._U(s2), self.logpi, self.lpw, d)
        a = post["alpha"][:, None, :]
        Eb = a * post["Eb"]
        term1 = _vinv_dot(a * post["Eb_g"], post["r_g"], self.rho)
        KL = -post["lbf_model"] + term1 - 0.5 * (post["alpha"] * post["tr_g"]).sum(1)
        quad = (post["alpha"] * post["tr"]).sum(1)                 # β-space E[βᵀV^{-1}β] for the ELBO / η terms
        smass = (a * post["wst"]).sum(-1)
        return post, Eb, quad, KL, s2, smass, a * post["e2"]

    def _noshare_now(self):
        """(B,) bool: rows whose current sweep runs with the shared state disabled (multi-start, P3-5(1))."""
        return self.niter < self.noshare

    def sweep(self):
        act = self.active
        ns = self._noshare_now()
        logpi_full = None
        if bool((ns & act).any()):
            # 'no-shared-first' start: π_both = 0, the other states renormalised (log π_s − log(1 − π_both))
            logpi_full = self.logpi
            lp = self.logpi - torch.log1p(-torch.exp(self.logpi[:, 3:4]).clamp_max(1 - 1e-12))
            lp[:, 3] = -math.inf
            self.logpi = torch.where(ns[:, None], lp, self.logpi)
        try:
            self._sweep(act)
        finally:
            if logpi_full is not None:
                self.logpi = logpi_full

    def _sweep(self, act):
        for l in range(self.cfg.L):
            RBminus = self.RBsum - self.RB[:, l]
            Ebsum = self.Eb.sum(1)
            Ebminus = Ebsum - self.Eb[:, l]
            r = self.z - self.Rt_from(RBminus, Ebminus)
            post, Eb, quad, KL, s2, smass, e2 = self.ser(r, self.s2[:, l])
            RBl = self.R_mv(Eb)
            a3 = act[:, None, None]
            self.alpha[:, l] = torch.where(act[:, None], post["alpha"], self.alpha[:, l])
            self.Eb[:, l] = torch.where(a3, Eb, self.Eb[:, l])
            self.rlast[:, l] = torch.where(a3, r, self.rlast[:, l])
            self.RB[:, l] = torch.where(a3, RBl, self.RB[:, l])
            self.RBsum = torch.where(a3, RBminus + RBl, self.RBsum)
            self.quad[:, l] = torch.where(act, quad, self.quad[:, l])
            self.KL[:, l] = torch.where(act, KL, self.KL[:, l])
            self.lbf[:, l] = torch.where(act, post["lbf_model"], self.lbf[:, l])
            self.s2[:, l] = torch.where(act[:, None], s2, self.s2[:, l])
            self.smass[:, l] = torch.where(act[:, None], smass, self.smass[:, l])
            if self.dvar is not None:
                self.E2[:, l] = torch.where(a3, e2, self.E2[:, l])
                self.dlast[:, l] = torch.where(a3, self.dvar, self.dlast[:, l])

    def _A_terms(self):
        """A_R = E[bᵀ(V^{-1}⊗R)b], A_I = E[bᵀ(V^{-1}⊗I)b] and the cross term E[b]ᵀ(V^{-1}⊗I)z."""
        Ebs = self.Eb.sum(1)
        rho_ = self.rho[:, None, None]
        den = 1.0 - self.rho * self.rho

        def vd_l(a, b):   # Σ_l Σ_j a_ljᵀ V^{-1} b_lj for (B,L,2,P)
            return ((a[:, :, 0] * b[:, :, 0] - rho_ * (a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0])
                     + a[:, :, 1] * b[:, :, 1]).sum((1, 2)) / den)
        q = self.quad.sum(1)
        A_R = _vinv_dot(Ebs, self.RBsum, self.rho) - vd_l(self.Eb, self.RB) + q
        A_I = _vinv_dot(Ebs, Ebs, self.rho) - vd_l(self.Eb, self.Eb) + q
        cross = _vinv_dot(Ebs, self.z, self.rho)
        return A_R, A_I, cross

    def _log_pi_prior(self):
        """log Dirichlet(1 + κ π_ref) density of π up to a constant (MAP objective in 'eb' mode)."""
        if self.cfg.pi_mode != "eb" or self.cfg.pi_prior_strength <= 0:
            return torch.zeros_like(self.eta)
        k = self.cfg.pi_prior_strength * self.pi_ref
        lp = torch.where(k[None, :] > 0, k[None, :] * self.logpi, torch.zeros_like(self.logpi))
        return lp.sum(1)

    def elbo(self):
        A_R, A_I, cross = self._A_terms()
        e = self.eta
        eloglik = cross - 0.5 * ((1.0 - e) * A_R + e * A_I)
        if self.eig is not None:
            eloglik = eloglik + self._eta_terms(e)
        return eloglik - self.KL.sum(1) + self._log_pi_prior()

    def update_pi(self):
        """EM / MAP step for the per-locus state prior: π_s ∝ Σ_{used l} q_l(s) + κ π_ref,s. Not applied during
        the no-shared phase of the second start (π stays at its initial value there)."""
        if self.cfg.pi_mode != "eb":
            return
        upd = self.active & ~self._noshare_now()
        # effects with both prior scales 0 have q_l(s) = π (BF = 1) and zero KL for any π, so they follow π at no
        # cost: the joint (π, q_unused) update uses the state mass of the used effects only (one-step fixed point).
        used = (self.s2.max(2).values > 0).to(self.smass.dtype)                  # (B,L)
        new = (self.smass * used[..., None]).sum(1) + self.cfg.pi_prior_strength * self.pi_ref[None, :]
        new = new / new.sum(1, keepdim=True)
        self.logpi = torch.where(upd[:, None], torch.log(new), self.logpi)

    def update_eta(self):
        A_R, A_I, _ = self._A_terms()
        lo, hi = self.cfg.eta_bounds
        lo = max(lo, self.cfg.eta_floor)
        if not self.eig_exact:
            lo = max(lo, self.cfg.eta_min_fp32)
        old = self.eta.clamp_min(lo)
        if A_R.is_cuda and self.cfg.use_triton and eta_update_triton is not None and self.ld2 is None:
            new = eta_update_triton(self.eig_lam, self.eig_q, self.eig_mask, self.eig_qnull, self.eig_nnull,
                                    self.p_real, self.rho, A_R, A_I, old, self.active, (lo, hi))
        else:
            B = A_R.shape[0]

            def g(le):
                e = torch.exp(le)
                return -(self._eta_terms(e) - 0.5 * ((1.0 - e) * A_R + e * A_I))
            le = brent_fmin(g, math.log(lo), math.log(hi), B, A_R.device)
            new = torch.exp(le)
            better = g(le) <= g(torch.log(old))
            new = torch.where(better, new, old)
        self.eta = torch.where(self.active, new, self.eta)

    def select(self, idx):
        names = list(self._SEL)
        if self.dvar is not None:
            names += ["dvar", "dlast", "E2", "nref_e"] + (["maf"] if self.maf is not None else [])
        for name in names:
            setattr(self, name, getattr(self, name)[idx])
        if self.dvar is not None:
            self._ref_tab = dict(self._ref_tab, nidx=self._ref_tab["nidx"][idx])
        if self.eig is not None:
            for name in ["eig_lam", "eig_q", "eig_mask", "eig_qnull", "eig_nnull"]:
                setattr(self, name, getattr(self, name)[idx])
        il = idx.tolist()
        self.elbo_hist = [self.elbo_hist[i] for i in il]
        self.eta_hist = [self.eta_hist[i] for i in il]
        self.ld = self.ld.select(idx)
        if self.ld2 is not None:
            self.ld2 = self.ld2.select(idx)

    def assign(self, slots, other: "_JointBucket"):
        """Continuous batching (M3.5): load the freshly initialised loci of `other` (same P, same kind) into the
        free working-set rows `slots` (list of ints, len = other.B)."""
        sl = torch.as_tensor(slots, device=self.z.device)
        names = list(self._SEL)
        if self.dvar is not None:
            names += ["dvar", "dlast", "E2", "nref_e"] + (["maf"] if self.maf is not None else [])
        for name in names:
            getattr(self, name)[sl] = getattr(other, name)
        if self.dvar is not None:
            self._ref_tab["nidx"][sl] = other._ref_tab["nidx"]
        if self.eig is not None:
            K, Ko = self.eig_lam.shape[1], other.eig_lam.shape[1]
            if Ko > K:
                raise RuntimeError("eigen width of refill exceeds the working set (bucket n_ref max)")
            for name in ["eig_lam", "eig_q", "eig_mask"]:
                buf = getattr(self, name)
                buf[sl] = 0
                buf[sl, :Ko] = getattr(other, name)
            self.eig_qnull[sl] = other.eig_qnull
            self.eig_nnull[sl] = other.eig_nnull
        for mine, theirs in ((self.ld, other.ld), (self.ld2, other.ld2)):
            if mine is None:
                continue
            n_o = theirs.mat.shape[1]
            if mine.kind == "dense":
                mine.mat[sl] = theirs.mat
            else:
                mine.mat[sl] = 0
                mine.mat[sl, :n_o] = theirs.mat
                mine.n_ref[sl] = theirs.n_ref
        for k, s_ in enumerate(slots):
            self.elbo_hist[s_] = []
            self.eta_hist[s_] = []

    # ---- final outputs ---------------------------------------------------------------------------------
    def outputs(self, rows):
        """Per-locus results for working-set rows (list of ints)."""
        cfg = self.cfg
        ri = torch.as_tensor(rows, device=self.z.device)
        L = cfg.L
        sub = lambda x: x[ri]
        z, rho, logpi, lpw = sub(self.z), sub(self.rho), sub(self.logpi), sub(self.lpw)
        eta = sub(self.eta)
        rlast = sub(self.rlast)
        s2 = sub(self.s2)
        thr = sub(self.thr)
        tau1, tau2, w1, w2 = thr[:, 0:1], thr[:, 1:2], thr[:, 2:3], thr[:, 3:4]          # (Bn,1)
        dl = sub(self.dlast) if self.dvar is not None else None
        Bn = len(rows)
        P = z.shape[2]
        f64 = torch.float64
        a_l = torch.zeros((Bn, L), dtype=f64, device=z.device)
        w_l = torch.zeros_like(a_l); p1_l = torch.zeros_like(a_l); p2_l = torch.zeros_like(a_l)
        pw1_l = torch.zeros_like(a_l); pw2_l = torch.zeros_like(a_l)
        state_prob = torch.zeros((Bn, L, 4), dtype=f64, device=z.device)
        alpha_all = torch.zeros((Bn, L, P), dtype=f64, device=z.device)
        # per variable: P(trait-t effect with own |β_t| >= weak_t), per effect (exclusion probabilities)
        q_weak = torch.zeros((Bn, L, 2, P), dtype=f64, device=z.device)
        pip_t = torch.zeros((Bn, 2, P), dtype=f64, device=z.device)
        T1, T2, W1, W2 = (x.expand(Bn, P) for x in (tau1, tau2, w1, w2))
        n_biv = 0
        for l in range(L):
            r = rlast[:, l]                # q(b_l) exactly as fitted by IBSS (residual at its last update)
            U = self._U(s2[:, l])
            post = _posterior(r, rho, U, logpi, lpw, None if dl is None else dl[:, l])
            al, wst, mu, Sig = post["alpha"], post["wst"], post["mu"], post["Sig"]
            Sig = Sig.expand(-1, -1, -1, P)                     # (Bn,3 states,3 entries,P)
            alpha_all[:, l] = al
            state_prob[:, l] = (al[:, None, :] * wst).sum(-1)
            # marginal noncentrality of the other effects at j: m = z̃ - r_l = R̃ E[b_{-l}]
            m = (z - r) if cfg.power_marginal else torch.zeros_like(z)
            m1, m2 = m[:, 0], m[:, 1]
            sp1, sp2 = powered_spec(m1, T1), powered_spec(m2, T2)
            # state 1-only: β1 ~ N(mu[:,0,0], Sig[:,0,0]); state 2-only: β2 ~ N(mu[:,1,1], Sig[:,1,2])
            sd_s1 = Sig[:, 0, 0].clamp_min(0).sqrt()
            sd_s2 = Sig[:, 1, 2].clamp_min(0).sqrt()
            pt1_s1 = set_prob(mu[:, 0, 0], sd_s1, sp1)
            pt2_s2 = set_prob(mu[:, 1, 1], sd_s2, sp2)
            ow1_s1 = marginal(mu[:, 0, 0], sd_s1, "out", W1)          # own |β1| >= w1 (exclusion)
            ow2_s2 = marginal(mu[:, 1, 1], sd_s2, "out", W2)
            # state both
            mb1, mb2 = mu[:, 2, 0], mu[:, 2, 1]
            S11, S12, S22 = Sig[:, 2, 0], Sig[:, 2, 1], Sig[:, 2, 2]
            sd1, sd2 = S11.clamp_min(0).sqrt(), S22.clamp_min(0).sqrt()
            pt1_s3 = set_prob(mb1, sd1, sp1)
            pt2_s3 = set_prob(mb2, sd2, sp2)
            ow1_s3 = marginal(mb1, sd1, "out", W1)
            ow2_s3 = marginal(mb2, sd2, "out", W2)
            w_s1, w_s2, w_s3 = wst[:, 1], wst[:, 2], wst[:, 3]
            wb = al * w_s3
            if cfg.nonweak_own:     # β = 0 a.s. when both prior scales are 0: every power event is exactly 0
                wb = wb * (s2[:, l].max(1).values > 0)[:, None]
            # bivariate events of the shared state. Entries are evaluated only where α·P(both|j) times a 1-D upper
            # bound of the event exceeds bvn_gather_tol (total omitted mass <= L·p·tol); s11 = 0 is exact.
            nw1_s3 = set_prob(mb1, sd1, nonweak_spec(m1, W1, cfg.nonweak_own))
            nw2_s3 = set_prob(mb2, sd2, nonweak_spec(m2, W2, cfg.nonweak_own))
            pw_ = lambda a, t, w: powered_spec(a, t, cfg.power_marginal)
            nw_ = lambda a, t, w: nonweak_spec(a, w, cfg.nonweak_own)
            wk_ = lambda a, t, w: weak_spec(a, w, cfg.nonweak_own)
            specs = {"a": (pw_, pw_, torch.minimum(pt1_s3, pt2_s3)), "w": (nw_, nw_, torch.minimum(nw1_s3, nw2_s3)),
                     "p1": (pw_, wk_, pt1_s3), "p2": (wk_, pw_, pt2_s3)}
            deg = sd1 <= 1e-150
            ev = {}
            for k, (f1, f2, ub) in specs.items():
                e_k = torch.zeros_like(wb)
                cand = wb * ub > cfg.bvn_gather_tol
                sd_ = torch.nonzero(cand & deg, as_tuple=True)
                if sd_[0].numel():
                    g = lambda x: x[sd_]
                    e_k[sd_] = (set_prob(g(mb1), torch.zeros_like(g(mb1)), f1(g(m1), g(T1), g(W1)))
                                * set_prob(g(mb2), g(sd2), f2(g(m2), g(T2), g(W2))))
                sel = torch.nonzero(cand & ~deg, as_tuple=True)
                nsel = sel[0].numel()
                n_biv += nsel
                ch = 100_000
                for c0 in range(0, nsel, ch):
                    ix = (sel[0][c0:c0 + ch], sel[1][c0:c0 + ch])
                    g = lambda x: x[ix]
                    e_k[ix] = bvn_set_prob(g(mb1), g(mb2), g(S11), g(S22), g(S12), f1(g(m1), g(T1), g(W1)),
                                           f2(g(m2), g(T2), g(W2)))
                ev[k] = e_k
            a_l[:, l] = (wb * ev["a"]).sum(1)
            w_l[:, l] = (wb * ev["w"]).sum(1)
            p1_l[:, l] = (al * w_s1 * pt1_s1).sum(1) + (wb * ev["p1"]).sum(1)
            p2_l[:, l] = (al * w_s2 * pt2_s2).sum(1) + (wb * ev["p2"]).sum(1)
            pw1_l[:, l] = (al * (w_s1 * pt1_s1 + w_s3 * pt1_s3)).sum(1)       # P(powered in trait 1)
            pw2_l[:, l] = (al * (w_s2 * pt2_s2 + w_s3 * pt2_s3)).sum(1)
            q_weak[:, l, 0] = al * (w_s1 * ow1_s1 + w_s3 * ow1_s3)
            q_weak[:, l, 1] = al * (w_s2 * ow2_s2 + w_s3 * ow2_s3)
            pip_t[:, 0] = 1 - (1 - pip_t[:, 0]) * (1 - al * (w_s1 + w_s3))
            pip_t[:, 1] = 1 - (1 - pip_t[:, 1]) * (1 - al * (w_s2 + w_s3))
        p12_l = p1_l + p2_l
        one = torch.ones_like(a_l)
        P_S = 1 - torch.prod(one - a_l, 1)
        P_D = (torch.prod(one - w_l, 1) - torch.prod(one - w_l - p1_l, 1) - torch.prod(one - w_l - p2_l, 1)
               + torch.prod(one - w_l - p12_l, 1)).clamp(0.0, 1.0)
        out = []
        A = alpha_all.cpu().numpy()
        arrs = {k: v.cpu().numpy() for k, v in dict(a=a_l, w=w_l, p1=p1_l, p2=p2_l, pw1=pw1_l, pw2=pw2_l,
                                                     state_prob=state_prob, s2=s2, P_S=P_S, P_D=P_D, eta=eta,
                                                     pip1=pip_t[:, 0], pip2=pip_t[:, 1], pi=torch.exp(logpi),
                                                     thr=thr).items()}
        qw = q_weak.cpu().numpy()
        for k, row in enumerate(rows):
            p = int(self.p_real[row])
            alpha = A[k, :, :p]
            act_l = arrs["s2"][k].max(1) > cfg.prior_tol
            pip = 1.0 - np.prod(1.0 - alpha[act_l], axis=0) if act_l.any() else np.zeros(p)
            # exclusion probabilities
            E = []
            for t, other in ((0, 1), (1, 0)):           # E_{other+1} uses CS of trait t+1
                pw = arrs["pw1"][k] if t == 0 else arrs["pw2"][k]
                mask = np.zeros(p, dtype=bool)
                for l in range(L):
                    if pw[l] >= cfg.powered_cs_min:
                        srt = np.argsort(-alpha[l], kind="stable")
                        n_in = min(int((np.cumsum(alpha[l][srt]) < cfg.coverage).sum()) + 1, p)
                        mask[srt[:n_in]] = True
                if not mask.any():
                    E.append(math.nan)
                else:
                    q = qw[k, :, other, :p][:, mask].sum(1)
                    E.append(float(np.prod(1.0 - np.clip(q, 0, 1))))
            E2, E1 = E[0], E[1]
            PS, PD = float(arrs["P_S"][k]), float(arrs["P_D"][k])
            res = dict(P_S=PS, P_D=PD, P_W=max(0.0, 1.0 - PS - PD), s=(PS / (PS + PD) if PS + PD > 0 else 0.5),
                       E1=E1, E2=E2, a=arrs["a"][k], w=arrs["w"][k], p1t=arrs["p1"][k], p2t=arrs["p2"][k],
                       pw1=arrs["pw1"][k], pw2=arrs["pw2"][k], state_prob=arrs["state_prob"][k], s2=arrs["s2"][k],
                       eta=float(arrs["eta"][k]), pi=arrs["pi"][k], thresholds=arrs["thr"][k], pip=pip,
                       pip1=arrs["pip1"][k][:p], pip2=arrs["pip2"][k][:p], alpha=alpha)
            out.append(res)
        return out


def _pad_factor(lds, n_pad):
    return [X if X.shape[0] == n_pad else np.concatenate([np.asarray(X), np.zeros((n_pad - X.shape[0], X.shape[1]),
                                                                                    dtype=np.asarray(X).dtype)])
            for X in lds]


def _make_state(items, kind, P, cfg: JointConfig, device, n_pad=None, n_pad2=None) -> _JointBucket:
    """Initialise a working set for `items` (dicts: z1, z2, ld, n, rho, n_ref, thr, nref_e, maf, optional ld2 /
    n_ref2 (per-trait LD; all items of a working set have it or none do) and noshare); width P; for factor LD the
    genotype rows are padded to n_pad (zero rows do not change XᵀX)."""
    B = len(items)
    f64 = torch.float64
    z = torch.zeros((B, 2, P), dtype=f64, device=device)
    valid = torch.zeros((B, P), dtype=torch.bool, device=device)
    p_real = torch.zeros(B, dtype=torch.int64, device=device)
    for i, it in enumerate(items):
        p = len(it["z1"])
        z[i, 0, :p] = torch.as_tensor(np.nan_to_num(np.asarray(it["z1"], float)), device=device)
        z[i, 1, :p] = torch.as_tensor(np.nan_to_num(np.asarray(it["z2"], float)), device=device)
        valid[i, :p] = True
        p_real[i] = p
    n = None
    if cfg.z_adjust and items[0]["n"] is not None:
        n = torch.as_tensor(np.asarray([it["n"] for it in items], dtype=np.float64), device=device)   # (B,2)
    lds = [it["ld"] for it in items]
    if kind == "factor" and n_pad is not None:
        lds = _pad_factor(lds, n_pad)
    ld = build_batch_ld(lds, kind, P, cfg.ld_dtype, device)
    if kind == "factor":
        ld.n_ref = torch.as_tensor(np.asarray([it["n_ref"] for it in items], dtype=np.float64), device=device)
    ld2 = None
    has2 = [it.get("ld2") is not None for it in items]
    if any(has2):
        if not all(has2):
            raise ValueError("per-trait LD: a working set must have ld2 for all loci or none")
        lds2 = [it["ld2"] for it in items]
        if kind == "factor" and n_pad2 is not None:
            lds2 = _pad_factor(lds2, n_pad2)
        ld2 = build_batch_ld(lds2, kind, P, cfg.ld_dtype, device)
        if kind == "factor":
            ld2.n_ref = torch.as_tensor(np.asarray([it["n_ref2"] for it in items], dtype=np.float64), device=device)
    noshare = [int(it.get("noshare", 0)) for it in items]
    rho = torch.as_tensor(np.asarray([it["rho"] for it in items], dtype=np.float64), device=device)
    thr_t = torch.as_tensor(np.asarray([it["thr"] for it in items], dtype=np.float64), device=device)
    nref_t = (torch.as_tensor(np.asarray([it["nref_e"] for it in items], dtype=np.float64), device=device)
              if cfg.ref_var else None)
    maf_t = None
    if cfg.ref_var and all(it["maf"] is not None for it in items):
        maf_t = torch.full((B, P), 0.5, dtype=f64, device=device)
        for i, it in enumerate(items):
            mm = np.asarray(it["maf"], float)
            mm = np.minimum(mm, 1.0 - mm)
            maf_t[i, :len(mm)] = torch.as_tensor(np.nan_to_num(mm, nan=0.5), device=device)
    return _JointBucket(z, valid, p_real, n, ld, rho, cfg, thr=thr_t, nref_e=nref_t, maf=maf_t, ld2=ld2,
                        noshare=noshare)


def _fit_stream(items, kind, P, cfg: JointConfig, device, slots: int, verbose=False):
    """IBSS for a group of loci of similar p with continuous batching (M3.5): a working set of `slots` loci;
    converged loci are flushed and their rows refilled from the queue (in chunks of >= refill_fraction · slots),
    so stragglers no longer hold an almost empty batch. Results are identical to fitting each locus alone
    (all updates are per locus)."""
    N = len(items)
    results = [None] * N
    scfg = SusieRSSConfig(coverage=cfg.coverage, min_abs_corr=cfg.min_abs_corr)
    n_pad = max(np.asarray(it["ld"]).shape[0] for it in items) if kind == "factor" else None
    n_pad2 = (max(np.asarray(it["ld2"]).shape[0] for it in items)
              if kind == "factor" and items[0].get("ld2") is not None else None)
    if cfg.ref_var and cfg.convergence != "alpha":
        raise ValueError("layer E (ref_var) requires convergence='alpha' (its SER updates do not maximise the ELBO)")
    slots = max(1, min(slots, N))
    queue = list(range(N))
    first, queue = queue[:slots], queue[slots:]
    st = _make_state([items[i] for i in first], kind, P, cfg, device, n_pad, n_pad2)
    st.orig = torch.as_tensor(first, device=st.z.device)
    occupied = torch.ones(len(first), dtype=torch.bool, device=st.z.device)

    def flush(rows):
        if not rows:
            return
        outs = st.outputs(rows)
        for k, r in enumerate(rows):
            o = int(st.orig[r])
            res = outs[k]
            p = int(st.p_real[r])
            Vproxy = res["s2"].max(1)
            res["sets"] = _credible_sets(res["alpha"], Vproxy, st.ld, r, p, scfg)
            res.update(elbo=np.array(st.elbo_hist[r]), eta_hist=np.array(st.eta_hist[r]), niter=int(st.niter[r]),
                       converged=bool(st.converged[r]), status="ok")
            results[o] = res

    refill_min = max(1, int(math.ceil(cfg.refill_fraction * slots)))
    while True:
        a_prev = st.alpha.clone() if cfg.convergence in ("alpha", "rel") else None
        st.sweep()
        act = st.active
        el = st.elbo()
        st.niter = st.niter + act.to(st.niter.dtype)
        el_c, actl, eta_c = el.cpu().tolist(), act.cpu().tolist(), st.eta.cpu().tolist()
        for r in range(len(actl)):
            if actl[r]:
                st.elbo_hist[r].append(el_c[r])
                st.eta_hist[r].append(eta_c[r])
        d = el - st.elbo_prev
        # multi-start: the sweeps of the no-shared phase (and the first full sweep after it, for the α rules)
        # fit a restricted model and never count as converged; the ELBO trace restarts after the phase.
        in_ns = st.niter <= st.noshare
        past = st.niter > st.noshare + 1
        if cfg.convergence == "susie016":
            conv = act & (d >= 0) & (d < cfg.tol) & (st.niter > 1)
        elif cfg.convergence == "alpha":
            conv = act & ((st.alpha - a_prev).abs().flatten(1).max(1).values < cfg.tol_alpha) & (st.niter > 1) & past
        elif cfg.convergence == "rel":
            da = (st.alpha - a_prev).abs().flatten(1).max(1).values
            # relative scale: the ELBO without the Dirichlet log-prior of π (an additive term whose size grows
            # with κ and carries no information about the fit's scale)
            scale = (el - st._log_pi_prior()).abs()
            conv = act & ((d < cfg.tol) | (d.abs() < cfg.tol_rel * scale) | ((da < cfg.tol_alpha) & (st.niter > 1) & past))
        elif cfg.convergence == "susie014":
            conv = act & (d < cfg.tol)
        else:
            raise ValueError(f"unknown convergence rule {cfg.convergence!r}")
        conv = conv & ~in_ns
        st.converged = st.converged | conv
        st.elbo_prev = torch.where(act, torch.where(in_ns, torch.full_like(el, -math.inf), el), st.elbo_prev)
        st.active = act & ~conv & torch.isfinite(el) & (st.niter < cfg.max_iter)
        if bool(st.active.any()):
            st.update_pi()
            if cfg.estimate_eta:
                st.update_eta()
            if cfg.ref_var:
                st.update_delta()
        done = occupied & ~st.active
        n_done = int(done.sum())
        if queue and n_done >= min(refill_min, len(queue)):
            rows = torch.nonzero(done).flatten().tolist()
            flush(rows)
            take, queue = queue[:len(rows)], queue[len(rows):]
            rows = rows[:len(take)]
            new = _make_state([items[i] for i in take], kind, P, cfg, device, n_pad, n_pad2)
            st.assign(rows, new)
            st.orig[torch.as_tensor(rows, device=st.z.device)] = torch.as_tensor(take, device=st.z.device)
            occupied[:] = True
            free = torch.nonzero(occupied & ~st.active & ~torch.isin(torch.arange(len(occupied), device=occupied.device),
                                                                      torch.as_tensor(rows, device=occupied.device))).flatten()
            if free.numel():                     # rows flushed but not refilled (queue exhausted)
                occupied[free] = False
            del new
            continue
        n_act = int(st.active.sum())
        if n_act == 0:
            break
        if not queue and n_act < cfg.compact_fraction * st.active.numel():
            flush(torch.nonzero(occupied & ~st.active).flatten().tolist())
            keep = torch.nonzero(st.active).flatten()
            st.select(keep)
            occupied = occupied[keep]
    flush([r for r in range(st.active.numel()) if bool(occupied[r]) and results[int(st.orig[r])] is None])
    missing = [i for i in range(N) if results[i] is None]
    if missing:
        raise RuntimeError(f"internal error: {len(missing)} loci without results")
    return results


def joint_susie_batch(z1s, z2s, lds, n1s=None, n2s=None, rhos=None, ld_kind="dense", cfg: JointConfig | None = None,
                      device=None, n_refs=None, verbose=False, types=None, ref_sizes=None, mafs=None,
                      lds2=None, n_refs2=None) -> list[dict]:
    """Fit JointSuSiE to many loci. z1s/z2s: lists of (p_i,) z-scores; lds: dense R or factor X per locus;
    n1s/n2s: per-locus sample sizes (for the z adjustment); rhos: per-locus residual correlation (rho_hat);
    types: per-locus (trait1_type, trait2_type) in {'gwas', 'mol'} (default cfg.default_types);
    ref_sizes: per-locus reference size n_ref (layer E); mafs: per-locus reference MAF arrays (layer E, optional);
    lds2 / n_refs2: optional per-locus trait-2 LD (same kind as lds; None entries = one LD for both traits), used when
    cfg.per_trait_ld (P3-5(2)); cfg.multistart fits every locus from two starts in the same batches and keeps the
    fit with the higher final ELBO (P3-5(1)).
    Loci are grouped by kind and p (p_max <= bucket_ratio · p_min); each group is fitted with a working set of at
    most mem_budget_bytes / (per-locus bytes) and max_batch loci, refilled continuously."""
    cfg = cfg or JointConfig()
    check_ld_dtype(cfg.ld_dtype)
    device = device or default_device()
    if torch.device(device).type == "cuda":
        assert_no_tf32()
    N = len(z1s)
    kinds = ld_kind if isinstance(ld_kind, list) else [ld_kind] * N
    rhos = rhos if rhos is not None else [0.0] * N
    n_refs = n_refs if n_refs is not None else [None] * N
    types = types if types is not None else [tuple(cfg.default_types)] * N
    thr_all = thresholds_for_types(types, cfg)
    if cfg.ref_var:
        if ref_sizes is None:
            ref_sizes = [n_refs[i] if n_refs[i] is not None else (lds[i].shape[0] if kinds[i] == "factor" else None)
                         for i in range(N)]
        if any(x is None for x in ref_sizes):
            raise ValueError("layer E (ref_var) needs ref_sizes (n_ref) for dense-LD loci")
    mafs = mafs if mafs is not None else [None] * N
    ns = [None] * N if (n1s is None or n2s is None) else [(float(a), float(b)) for a, b in zip(n1s, n2s)]
    lds2 = lds2 if (lds2 is not None and cfg.per_trait_ld) else [None] * N
    n_refs2 = n_refs2 if n_refs2 is not None else [None] * N
    for i in range(N):
        if lds2[i] is not None and np.asarray(lds2[i]).shape[-1] != np.asarray(lds[i]).shape[-1]:
            raise ValueError(f"locus {i}: trait-2 LD has a different number of variants")
    bytes_per = 4 if cfg.ld_dtype == torch.float32 else 8

    def mk(i, noshare):
        f = kinds[i] == "factor"
        return dict(z1=z1s[i], z2=z2s[i], ld=lds[i], n=ns[i], rho=rhos[i],
                    n_ref=(n_refs[i] if n_refs[i] is not None else lds[i].shape[0]) if f else None,
                    thr=thr_all[i], nref_e=ref_sizes[i] if cfg.ref_var else None, maf=mafs[i], ld2=lds2[i],
                    n_ref2=((n_refs2[i] if n_refs2[i] is not None else lds2[i].shape[0]) if f and lds2[i] is not None
                            else None), noshare=noshare)
    starts = [0] + ([int(cfg.noshare_sweeps)] if cfg.multistart and cfg.noshare_sweeps > 0 else [])
    NS = len(starts)
    items = [mk(i, ns_) for ns_ in starts for i in range(N)]       # job k = start k // N, locus k % N
    out_all = [None] * (N * NS)
    out = out_all
    groups = {}
    for k in range(N * NS):
        i = k % N
        groups.setdefault((kinds[i], ns[i] is None, lds2[i] is not None), []).append(k)
    for (kind, _, two), idxs in groups.items():
        idxs = sorted(idxs, key=lambda k: (len(z1s[k % N]), k % N, k // N))   # both starts of a locus adjacent
        bucket, p_min = [], None

        def run(bucket):
            if not bucket:
                return
            P = max(len(z1s[k % N]) for k in bucket)
            nld = max(np.asarray(lds[k % N]).shape[0] for k in bucket) if kind == "factor" else P
            if two:
                nld += max(np.asarray(lds2[k % N]).shape[0] for k in bucket) if kind == "factor" else P
            per = nld * P * bytes_per + 120 * cfg.L * P * 8
            slots = int(max(1, min(cfg.max_batch, cfg.mem_budget_bytes // per)))
            t0 = time.time()
            res = _fit_stream([items[i] for i in bucket], kind, P, cfg, device, slots, verbose)
            if verbose:
                print(f"joint group kind={kind} N={len(bucket)} slots={slots} P={P} {time.time() - t0:.2f}s", flush=True)
            for i, r in zip(bucket, res):
                out[i] = r

        for k in idxs:
            p = len(z1s[k % N])
            if bucket and p > cfg.bucket_ratio * p_min:
                run(bucket)
                bucket = []
            if not bucket:
                p_min = p
            bucket.append(k)
        run(bucket)
    if NS == 1:
        for r in out_all:
            r.update(start="default", elbo_starts=[_final_elbo(r)], per_trait_ld=None)
        res = out_all
    else:
        res = []
        for i in range(N):
            a, b = out_all[i], out_all[N + i]
            ea, eb = _final_elbo(a), _final_elbo(b)
            pick_b = math.isfinite(eb) and (not math.isfinite(ea) or eb > ea)
            r = b if pick_b else a
            r.update(start="noshare" if pick_b else "default", elbo_starts=[ea, eb])
            res.append(r)
    for i, r in enumerate(res):
        r["per_trait_ld"] = lds2[i] is not None
    return res


def _final_elbo(r) -> float:
    e = r.get("elbo")
    return float(e[-1]) if e is not None and len(e) else -math.inf

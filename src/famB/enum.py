"""Exact enumeration engine for ExactColoc (DESIGN v3 Sec. 5 B).

Model (trait-major stacking, z = [z1; z2] over the screened union u of M variants):
    z ~ N((I2 (x) Rt) beta, V (x) Rt),   Rt = (1 - eta) R_uu + eta I,   V = [[1, rho], [rho, 1]]
A *configuration* is a pair (S1, S2) of variant sets, |S_t| <= Kc (Kc = 1 or 2). Its active coordinates
are the (trait, variant) pairs x in s = {(1, j): j in S1} u {(2, j): j in S2}; beta_s ~ N(0, D).

Support-space Bayes factor (exact; derivation: X = I2 (x) Rt, Sigma = V (x) Rt => X' Sigma^-1 = V^-1 (x) I):
    Phi = V^-1 (x) Rt,  b = (V^-1 (x) I) z   (both built over u for BOTH traits, then restricted to s)
    BF  = |I + D Phi_ss|^(-1/2) exp( 1/2 b_s' (I + D Phi_ss)^-1 D b_s )
We evaluate it in the symmetric form with D = L L' (L lower-triangular, rank-deficient allowed):
    A = I + L' Phi_ss L  (SPD, eigenvalues >= 1),  y = L' b_s,
    log BF = -1/2 log|A| + 1/2 y' A^-1 y
which is identical (Sylvester / push-through identities) and is stable for rank-1 shared priors and
near-singular Rt. Each configuration needs one 4x4 Cholesky, done in closed form (unrolled).

Coordinates are ordered [ (1, a1), (1, b1), (2, a2), (2, b2) ]; absent coordinates have L rows = 0, which
makes them contribute exactly nothing (A = I on them, y = 0).

Prior covariance D: effects of different variants are independent; a variant active in both traits has
cov = rho_b * sd_1 * sd_2 (rho_b = 1 gives the rank-1 shared prior). Per (trait, variant) the prior sd
may be a K-component scale mixture: each configuration with k active coordinates then expands into K^k
*terms* (one per component assignment), each a Gaussian prior; the sums run over all terms.

Configuration-prior: log pi(config) = logprior_type[n_sh, n_1only, n_2only] + sum_{j in S1 u S2} log m_j
(+ sum of the log mixture weights of the active coordinates' components), where m_j is the duplicate
multiplicity of screened representative j (Sec. 5 B screen).

Sums over configurations/terms are accumulated in fp64 (logsumexp per hypothesis category), in a
streaming two-pass scheme: pass 1 gives the category log-normalisers; pass 2 gives PIPs and draws
`n_samples` terms with probability proportional to their posterior mass (stratified systematic sampling),
which feed the Monte-Carlo oracle-event posteriors (events.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

TX = (0, 0, 1, 1)  # trait (0-based) of coordinate slot x


def assert_precision_guards() -> None:
    """DESIGN Sec. 5 A/7.1(10): TF32 must be off (fp64 is unaffected, but we assert globally)."""
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 matmul must be disabled"


def config_table(M: int, Kc: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-trait configurations with <= Kc causal variants among M: arrays (a, b), -1 = absent.
    Index 0 is always the empty configuration; then the M singletons; then pairs a < b."""
    if Kc not in (1, 2):
        raise ValueError("Kc must be 1 or 2")
    a = [np.array([-1]), np.arange(M)]
    b = [np.array([-1]), np.full(M, -1)]
    if Kc == 2 and M >= 2:
        i, j = np.triu_indices(M, k=1)
        a.append(i)
        b.append(j)
    return np.concatenate(a).astype(np.int64), np.concatenate(b).astype(np.int64)


class TermSet:
    """Enumerates all terms (configuration pair x mixture-component assignment) for a given M.

    single_trait=True enumerates (S1, empty) only (used for per-trait empirical Bayes).
    Terms are addressed by a flat int64 id in [0, n_terms); `decode` maps ids to coordinates.
    """

    def __init__(self, M: int, Kc: int, K: int = 1, single_trait: bool = False, device="cpu"):
        self.M, self.Kc, self.K, self.single_trait = int(M), int(Kc), int(K), bool(single_trait)
        self.device = torch.device(device)
        ca, cb = config_table(M, Kc)
        self.n_c = len(ca)
        self.n_c2 = 1 if single_trait else self.n_c
        self.n_pairs = self.n_c * self.n_c2
        self.ca = torch.as_tensor(ca, device=self.device)
        self.cb = torch.as_tensor(cb, device=self.device)
        size = (ca >= 0).astype(np.int64) + (cb >= 0).astype(np.int64)
        if K == 1:
            self.offsets = None
            self.n_terms = self.n_pairs
        else:
            k = (size[:, None] + size[None, : self.n_c2]).reshape(-1)
            cnt = np.power(np.int64(K), k)
            off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
            self.offsets = torch.as_tensor(off, device=self.device)
            self.n_terms = int(off[-1])
        self.powK = torch.as_tensor([K ** i for i in range(4)], dtype=torch.int64, device=self.device)

    def decode(self, t: torch.Tensor):
        """term ids [P] -> V [P,4] (variant index, 0 where absent), m [P,4] bool, kx [P,4] component."""
        if self.offsets is None:
            q = t
            combo = None
        else:
            q = torch.searchsorted(self.offsets, t, right=True) - 1
            combo = t - self.offsets[q]
        c1 = q // self.n_c2
        c2 = q % self.n_c2
        V = torch.stack([self.ca[c1], self.cb[c1], self.ca[c2], self.cb[c2]], dim=-1)
        m = V >= 0
        if combo is None:
            kx = torch.zeros_like(V)
        else:
            rank = (torch.cumsum(m.to(torch.int64), dim=-1) - 1).clamp(min=0)
            kx = torch.div(combo[:, None], self.powK[rank], rounding_mode="floor") % self.K
            kx = torch.where(m, kx, torch.zeros_like(kx))
        return V.clamp(min=0), m, kx


@dataclass
class LocusTensors:
    """Per-locus model inputs, batched over nB loci sharing the same M (all on one device).

    Rt            [nB, M, M] regularised LD of the union (None = identity; exact whenever only
                  diagonal entries enter, e.g. coloc mode: rho = 0, <= 1 causal per trait)
    b             [nB, 2, M]  (V^-1 (x) I) z_u
    Vinv          [nB, 2, 2]
    sd            [nB, 2, M, K] prior sd of the effect (z / noncentrality scale) per trait, variant, component
    logwk         [nB, 2, K]  log mixture weights per trait (0 for K = 1)
    rho_b         [nB]        prior correlation of a shared variant's two effects
    logm          [nB, M]     log duplicate multiplicity
    logprior_type [nB, 3, 3, 3] log prior by (n_shared, n_1only, n_2only); -inf = excluded
    eta           [nB]        regularisation used to build Rt (to recover raw r for the U class)
    """

    Rt: torch.Tensor | None
    b: torch.Tensor
    Vinv: torch.Tensor
    sd: torch.Tensor
    logwk: torch.Tensor
    rho_b: torch.Tensor
    logm: torch.Tensor
    logprior_type: torch.Tensor
    eta: torch.Tensor
    Phi: torch.Tensor | None = None  # [nB, 2M, 2M] general X' Sigma^-1 X (index t * M + v); None = Vinv (x) Rt.
                                     # Set by famB.mismatch (B-mm: Sigma = V (x) Rt + Delta, full-z_M likelihood);
                                     # b must then be X' Sigma^-1 z.
    extra: dict = field(default_factory=dict)

    @property
    def nB(self) -> int:
        return int(self.b.shape[0])

    @property
    def M(self) -> int:
        return int(self.b.shape[2])

    @property
    def K(self) -> int:
        return int(self.sd.shape[3])


def _shared_flags(V, m):
    a1, b1, a2, b2 = V.unbind(-1)
    m0, m1, m2, m3 = m.unbind(-1)
    s20 = m2 & m0 & (a2 == a1)
    s21 = m2 & m1 & (a2 == b1)
    s30 = m3 & m0 & (b2 == a1)
    s31 = m3 & m1 & (b2 == b1)
    return s20, s21, s30, s31


def structure(V, m):
    """(n_shared, n_1only, n_2only, category) per term; category indexes constants.CAT_NAMES."""
    s20, s21, s30, s31 = _shared_flags(V, m)
    nsh = (s20 | s21).to(torch.int64) + (s30 | s31).to(torch.int64)
    s1 = m[:, 0].to(torch.int64) + m[:, 1].to(torch.int64)
    s2 = m[:, 2].to(torch.int64) + m[:, 3].to(torch.int64)
    n1 = s1 - nsh
    n2 = s2 - nsh
    cat = torch.full_like(s1, 5)
    cat = torch.where((nsh > 0) & (n1 == 0) & (n2 == 0), torch.full_like(cat, 4), cat)
    cat = torch.where((s1 > 0) & (s2 > 0) & (nsh == 0), torch.full_like(cat, 3), cat)
    cat = torch.where((s1 == 0) & (s2 > 0), torch.full_like(cat, 2), cat)
    cat = torch.where((s1 > 0) & (s2 == 0), torch.full_like(cat, 1), cat)
    cat = torch.where((s1 == 0) & (s2 == 0), torch.zeros_like(cat), cat)
    return nsh, n1, n2, cat, (s20, s21, s30, s31)


def L_entries(sd, flags, rho_b):
    """Nonzero entries of the lower Cholesky factor of D for coordinates [1a, 1b, 2a, 2b].
    sd: list of 4 tensors (already masked); flags: (s20, s21, s30, s31) as float; rho_b broadcastable.
    Returns dict with keys L00 L11 L20 L21 L22 L30 L31 L33 (L10 = L32 = 0 exactly; see module doc)."""
    s20, s21, s30, s31 = flags
    sd0, sd1, sd2, sd3 = sd
    rb = rho_b
    one = torch.ones((), dtype=sd0.dtype, device=sd0.device)
    c2 = torch.sqrt(torch.clamp(one - rb * rb * (s20 + s21), min=0.0))
    c3 = torch.sqrt(torch.clamp(one - rb * rb * (s30 + s31), min=0.0))
    return dict(L00=sd0, L11=sd1, L20=rb * sd2 * s20, L21=rb * sd2 * s21, L22=sd2 * c2,
                L30=rb * sd3 * s30, L31=rb * sd3 * s31, L33=sd3 * c3)


def _gather_R(T: LocusTensors, V, x, y, dtype, bidx=None):
    if T.Rt is None:
        return (V[:, x] == V[:, y]).to(dtype)[None] if bidx is None else (V[:, x] == V[:, y]).to(dtype)
    if bidx is None:
        return T.Rt[:, V[:, x], V[:, y]]
    return T.Rt[bidx, V[:, x], V[:, y]]


def kernel(T: LocusTensors, V, m, kx, dtype=torch.float64):
    """log prior + log BF for terms (V, m, kx) [P, 4] at every locus of the batch -> logw [nB, P], cat [P].

    All arithmetic in `dtype` (fp64 default); 4x4 Cholesky unrolled."""
    nsh, n1, n2, cat, flags = structure(V, m)
    s20, s21, s30, s31 = flags
    mf = m.to(dtype)
    # ---- prior
    lp = T.logprior_type[:, nsh, n1, n2].to(torch.float64)  # [nB, P]
    for x in range(4):  # where(), not multiply: a zero-weight component (log w = -inf) must not give 0 * -inf
        lp = lp + torch.where(m[:, x][None], T.logwk[:, TX[x], kx[:, x]] + T.logm[:, V[:, x]],
                              torch.zeros((), dtype=torch.float64, device=lp.device))
    shared2 = (s20 | s21).to(torch.float64)
    shared3 = (s30 | s31).to(torch.float64)
    lp = lp - shared2 * T.logm[:, V[:, 2]] - shared3 * T.logm[:, V[:, 3]]
    # ---- L (prior covariance factor)
    sdx = [T.sd[:, TX[x], V[:, x], kx[:, x]].to(dtype) * mf[:, x] for x in range(4)]
    fl = [f.to(dtype) for f in flags]
    rb = T.rho_b.to(dtype)[:, None]
    Le = L_entries(sdx, fl, rb)
    L00, L11, L20, L21, L22, L30, L31, L33 = (Le[k] for k in ("L00", "L11", "L20", "L21", "L22", "L30", "L31", "L33"))
    # ---- Phi_xy = Vinv[t_x, t_y] * Rt[v_x, v_y]
    Vi = T.Vinv.to(dtype)
    P_ = {}
    if T.Phi is not None:
        M_ = T.M
        gi = [TX[x] * M_ + V[:, x] for x in range(4)]
    for x in range(4):
        for y in range(x, 4):
            if T.Phi is not None:
                P_[(x, y)] = T.Phi[:, gi[x], gi[y]].to(dtype)
            else:
                P_[(x, y)] = Vi[:, TX[x], TX[y]][:, None] * _gather_R(T, V, x, y, dtype).to(dtype)
            P_[(y, x)] = P_[(x, y)]
    bs = [T.b[:, TX[x], V[:, x]].to(dtype) for x in range(4)]
    # ---- H = Phi L (columns j)
    H = {}
    for x in range(4):
        H[(x, 0)] = P_[(x, 0)] * L00 + P_[(x, 2)] * L20 + P_[(x, 3)] * L30
        H[(x, 1)] = P_[(x, 1)] * L11 + P_[(x, 2)] * L21 + P_[(x, 3)] * L31
        H[(x, 2)] = P_[(x, 2)] * L22
        H[(x, 3)] = P_[(x, 3)] * L33
    del P_

    def G(i, j):  # (L' H)_ij
        if i == 0:
            return L00 * H[(0, j)] + L20 * H[(2, j)] + L30 * H[(3, j)]
        if i == 1:
            return L11 * H[(1, j)] + L21 * H[(2, j)] + L31 * H[(3, j)]
        if i == 2:
            return L22 * H[(2, j)]
        return L33 * H[(3, j)]

    a00 = 1.0 + G(0, 0); a01 = G(0, 1); a02 = G(0, 2); a03 = G(0, 3)
    a11 = 1.0 + G(1, 1); a12 = G(1, 2); a13 = G(1, 3)
    a22 = 1.0 + G(2, 2); a23 = G(2, 3)
    a33 = 1.0 + G(3, 3)
    del H
    y0 = L00 * bs[0] + L20 * bs[2] + L30 * bs[3]
    y1 = L11 * bs[1] + L21 * bs[2] + L31 * bs[3]
    y2 = L22 * bs[2]
    y3 = L33 * bs[3]
    # ---- unrolled Cholesky of A (SPD, diag >= 1)
    l00 = torch.sqrt(a00)
    l10 = a01 / l00
    l11 = torch.sqrt(a11 - l10 * l10)
    l20 = a02 / l00
    l21 = (a12 - l20 * l10) / l11
    l22 = torch.sqrt(a22 - l20 * l20 - l21 * l21)
    l30 = a03 / l00
    l31 = (a13 - l30 * l10) / l11
    l32 = (a23 - l30 * l20 - l31 * l21) / l22
    l33 = torch.sqrt(a33 - l30 * l30 - l31 * l31 - l32 * l32)
    w0 = y0 / l00
    w1 = (y1 - l10 * w0) / l11
    w2 = (y2 - l20 * w0 - l21 * w1) / l22
    w3 = (y3 - l30 * w0 - l31 * w1 - l32 * w2) / l33
    logdet = 2.0 * (torch.log(l00) + torch.log(l11) + torch.log(l22) + torch.log(l33))
    quad = w0 * w0 + w1 * w1 + w2 * w2 + w3 * w3
    logbf = (-0.5 * logdet + 0.5 * quad).to(torch.float64)
    return lp + logbf, cat


def eval_terms(T: LocusTensors, ts: TermSet, t: torch.Tensor, dtype=torch.float64):
    """decode + kernel for term ids t -> (logw [nB, P], cat [P], V, m)."""
    V, m, kx = ts.decode(t)
    logw, cat = kernel(T, V, m, kx, dtype=dtype)
    return logw, cat, V, m


_COMPILED = {}


def get_evaluator(compile: bool = False):
    """Eager or torch.compile'd (inductor, dynamic shapes) term evaluator. Compiled == eager to ~1e-13
    (tests/famB/test_engine.py); compilation costs ~10-30 s once per process (+ once per new nB pattern)."""
    if not compile:
        return eval_terms
    if "f" not in _COMPILED:
        import torch.fx.experimental._config as fxcfg
        fxcfg.use_duck_shape = False  # nB == 2 (traits) must not be specialised -> fewer recompiles
        _COMPILED["f"] = torch.compile(eval_terms, dynamic=True)
    return _COMPILED["f"]


def _chunks(n_terms: int, chunk: int):
    s = 0
    while s < n_terms:
        e = min(n_terms, s + chunk)
        yield s, e
        s = e


def enumerate_batch(T: LocusTensors, ts: TermSet, n_samples: int = 0, generator: torch.Generator | None = None,
                    chunk_elems: int = 1 << 22, cache_elems: int = 1 << 27, dtype=torch.float64,
                    want_pip: bool = True, evaluator=None):
    """Run the exact enumeration for a batch of loci sharing M.

    Returns dict:
      logZ_cat [nB, 6] (log sum of prior*BF per category, relative to the empty configuration having
               prior weight exp(logprior_type[0,0,0])), logZ [nB], PP [nB, 6],
      pip [nB, 2, M] marginal posterior inclusion per trait (exact on the rectangular path, Monte Carlo
               from the samples on the generic path),
      spip [nB, M] P(variant is a shared causal) (Monte Carlo from the samples on both paths),
      map_term [nB] (highest-mass term id), map_logw [nB],
      samples [nB, n_samples] term ids drawn proportional to posterior mass (if n_samples > 0).
    Single-component priors (K = 1) use the rectangular fast path (terms = c1-major grid of trait
    configurations: exact row/column marginals, two-stage sampling); mixtures use the generic path.
    """
    if ts.offsets is None:
        return _enumerate_rect(T, ts, n_samples, generator, chunk_elems, cache_elems, dtype, want_pip,
                               evaluator or eval_terms)
    return _enumerate_generic(T, ts, n_samples, generator, chunk_elems, cache_elems, dtype, want_pip, evaluator)


def _cat_sums(ew, cat):
    """[nB, 6] sums of ew per category via masked reductions (no atomics)."""
    return torch.stack([(ew * (cat == c).to(ew.dtype)[None]).sum(dim=1) for c in range(6)], dim=1)


def rand_rows(nB: int, tail: tuple, dev, generator, normal: bool = False) -> torch.Tensor:
    """Random numbers of shape [nB, *tail] (fp64). `generator` is one torch.Generator (draws depend on the batch
    composition) or a list of nB per-locus generators: row b is drawn from generator[b] only, so a locus gets the
    same draws whatever batch it is in (method.ExactColoc keys them by derive(seed, ..., locus_id))."""
    f = torch.randn if normal else torch.rand
    if isinstance(generator, (list, tuple)):
        if len(generator) != nB:
            raise ValueError(f"{len(generator)} generators for {nB} loci")
        return torch.stack([f(*tail, device=dev, dtype=torch.float64, generator=g) for g in generator])
    return f(nB, *tail, device=dev, dtype=torch.float64, generator=generator)


def _stratified_uniforms(nB, N, dev, generator):
    return (torch.arange(N, device=dev, dtype=torch.float64)[None] + rand_rows(nB, (N,), dev, generator)) / N


def _enumerate_rect(T, ts, n_samples, generator, chunk_elems, cache_elems, dtype, want_pip, ev):
    dev = T.b.device
    nB, M = T.nB, T.M
    n1, n2 = ts.n_c, ts.n_c2
    rows_per = max(1, chunk_elems // max(1, nB * n2))
    use_cache = n1 * n2 * nB <= cache_elems
    cache = torch.empty((nB, n1, n2), dtype=torch.float64, device=dev) if use_cache else None
    logZc = torch.full((nB, 6), -math.inf, dtype=torch.float64, device=dev)
    logq1 = torch.full((nB, n1), -math.inf, dtype=torch.float64, device=dev)
    logq2 = torch.full((nB, n2), -math.inf, dtype=torch.float64, device=dev)
    best = torch.full((nB,), -math.inf, dtype=torch.float64, device=dev)
    best_id = torch.zeros(nB, dtype=torch.int64, device=dev)
    for r0 in range(0, n1, rows_per):
        r1 = min(n1, r0 + rows_per)
        t = torch.arange(r0 * n2, r1 * n2, device=dev, dtype=torch.int64)
        logw, cat, _, _ = ev(T, ts, t, dtype)
        mx, am = logw.max(dim=1)
        upd = mx > best
        best = torch.where(upd, mx, best)
        best_id = torch.where(upd, t[am], best_id)
        mxs = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        ew = torch.exp(logw - mxs[:, None])
        sums = _cat_sums(ew, cat)
        logZc = torch.logaddexp(logZc, torch.log(sums) + mxs[:, None])
        lw3 = logw.view(nB, r1 - r0, n2)  # stable per-row / per-column log-sums (no underflow to -inf)
        logq1[:, r0:r1] = torch.logsumexp(lw3, dim=2)
        logq2 = torch.logaddexp(logq2, torch.logsumexp(lw3, dim=1))
        if use_cache:
            cache[:, r0:r1] = logw.view(nB, r1 - r0, n2)
    logZ = torch.logsumexp(logZc, dim=1)
    out = dict(logZ_cat=logZc, logZ=logZ, PP=torch.exp(logZc - logZ[:, None]), map_term=best_id, map_logw=best)
    if want_pip:
        q1 = torch.exp(logq1 - logZ[:, None])
        q2 = torch.exp(logq2 - logZ[:, None])
        pip = torch.zeros(nB, 2, M, dtype=torch.float64, device=dev)
        for c, q, tr in ((ts.ca[:n1], q1, 0), (ts.cb[:n1], q1, 0), (ts.ca[:n2], q2, 1), (ts.cb[:n2], q2, 1)):
            ok = c >= 0
            pip[:, tr].index_add_(1, c[ok], q[:, ok])
        out["pip"] = pip
    if n_samples <= 0:
        return out
    N = int(n_samples)
    # stage 1: rows (trait-1 configuration) by stratified sampling on the exact row marginals
    q1n = torch.exp(logq1 - logZ[:, None])
    C1 = torch.cumsum(q1n, dim=1)
    u = _stratified_uniforms(nB, N, dev, generator) * C1[:, -1:]
    rows = torch.searchsorted(C1.contiguous(), u, right=True).clamp(max=n1 - 1)  # [nB, N]
    # stage 2: column within the row, by inverse CDF of the row's weights
    u2 = rand_rows(nB, (N,), dev, generator)
    cols = torch.empty_like(rows)
    if use_cache:
        uniq, inv = torch.unique(rows, return_inverse=True)  # rows used by any locus
        for s in range(0, uniq.numel(), max(1, chunk_elems // max(1, nB * n2))):
            ur = uniq[s:s + max(1, chunk_elems // max(1, nB * n2))]
            _sample_cols(cache[:, ur], logq1[:, ur], ur, rows, u2, cols)
    else:
        uniq = torch.unique(rows)
        step = max(1, chunk_elems // max(1, nB * n2))
        for s in range(0, uniq.numel(), step):
            ur = uniq[s:s + step]
            t = (ur[:, None] * n2 + torch.arange(n2, device=dev)[None]).reshape(-1)
            logw, _, _, _ = ev(T, ts, t, dtype)
            _sample_cols(logw.view(nB, ur.numel(), n2), logq1[:, ur], ur, rows, u2, cols)
    samples = rows * n2 + cols
    out["samples"] = samples
    # Monte-Carlo shared PIP from the samples
    V, m, _ = ts.decode(samples.reshape(-1))
    _, _, _, _, (s20, s21, s30, s31) = structure(V, m)
    spip = torch.zeros(nB, M, dtype=torch.float64, device=dev)
    bidx = torch.arange(nB, device=dev).repeat_interleave(N)
    flat = bidx * M
    spip.view(-1).index_add_(0, flat + V[:, 2], (s20 | s21).to(torch.float64) / N)
    spip.view(-1).index_add_(0, flat + V[:, 3], (s30 | s31).to(torch.float64) / N)
    out["spip"] = spip
    return out


def _sample_cols(lw_rows, logq1_rows, ur, rows, u2, cols):
    """lw_rows [nB, k, n2] log weights of the (sorted, unique) rows ur; for every sample whose row is in
    ur draw its column by inverse CDF. Rows are laid end to end with integer offsets (row i occupies
    (i, i + 1]) so that one searchsorted over [nB, k * n2] serves all samples without per-sample copies."""
    nB, k, n2 = lw_rows.shape
    # each row normalised (sums to 1 up to rounding); rows with zero mass for a locus (possible when the
    # row was sampled only by another locus of the batch) get p = 0 so the concatenated CDF stays sorted
    p = torch.exp(lw_rows - logq1_rows[:, :, None])
    p = torch.where(torch.isfinite(logq1_rows)[:, :, None] & torch.isfinite(p), p, torch.zeros_like(p))
    C = torch.cumsum(p, dim=2)
    last = C[:, :, -1]  # [nB, k]
    C = C + torch.arange(k, device=C.device, dtype=C.dtype)[None, :, None]
    pos = torch.searchsorted(ur, rows.clamp(min=ur[0], max=ur[-1]))
    hit = (pos < k) & (ur[pos.clamp(max=k - 1)] == rows)
    pos = pos.clamp(max=k - 1)
    tgt = pos.to(C.dtype) + u2 * torch.gather(last, 1, pos)
    idx = torch.searchsorted(C.view(nB, k * n2).contiguous(), tgt, right=True)
    c = (idx - pos * n2).clamp(min=0, max=n2 - 1)
    cols.copy_(torch.where(hit, c, cols))


def _mc_pips(ts, samples, M):
    """Monte-Carlo per-trait PIPs and shared-causal PIPs from sampled term ids [nB, N]."""
    nB, N = samples.shape
    dev = samples.device
    V, m, _ = ts.decode(samples.reshape(-1))
    _, _, _, _, (s20, s21, s30, s31) = structure(V, m)
    bidx = torch.arange(nB, device=dev).repeat_interleave(N)
    pip = torch.zeros(nB * 2 * M, dtype=torch.float64, device=dev)
    for x in range(4):
        pip.index_add_(0, bidx * 2 * M + TX[x] * M + V[:, x], m[:, x].to(torch.float64) / N)
    spip = torch.zeros(nB * M, dtype=torch.float64, device=dev)
    spip.index_add_(0, bidx * M + V[:, 2], (s20 | s21).to(torch.float64) / N)
    spip.index_add_(0, bidx * M + V[:, 3], (s30 | s31).to(torch.float64) / N)
    return pip.view(nB, 2, M), spip.view(nB, M)


def _enumerate_generic(T: LocusTensors, ts: TermSet, n_samples: int = 0, generator: torch.Generator | None = None,
                       chunk_elems: int = 1 << 22, cache_elems: int = 1 << 27, dtype=torch.float64,
                       want_pip: bool = True, evaluator=None):
    """Generic path (mixture priors): pass 1 = category sums (cached logw if it fits), pass 2 = stratified
    systematic sampling of terms over the global cumulative mass. PIPs are Monte Carlo (from the samples)."""
    dev = T.b.device
    nB, M = T.nB, T.M
    ev = evaluator or eval_terms
    chunk = max(1024, chunk_elems // max(1, nB))
    use_cache = ts.n_terms * nB <= cache_elems
    cache = []
    logZc = torch.full((nB, 6), -math.inf, dtype=torch.float64, device=dev)
    best = torch.full((nB,), -math.inf, dtype=torch.float64, device=dev)
    best_id = torch.zeros(nB, dtype=torch.int64, device=dev)
    for s, e in _chunks(ts.n_terms, chunk):
        t = torch.arange(s, e, device=dev, dtype=torch.int64)
        logw, cat, _, _ = ev(T, ts, t, dtype)
        mx, am = logw.max(dim=1)
        upd = mx > best
        best = torch.where(upd, mx, best)
        best_id = torch.where(upd, t[am], best_id)
        mxs = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        sums = _cat_sums(torch.exp(logw - mxs[:, None]), cat)
        logZc = torch.logaddexp(logZc, torch.log(sums) + mxs[:, None])
        if use_cache:
            cache.append((s, e, logw))
    logZ = torch.logsumexp(logZc, dim=1)
    out = dict(logZ_cat=logZc, logZ=logZ, PP=torch.exp(logZc - logZ[:, None]), map_term=best_id, map_logw=best)
    if n_samples <= 0:
        return out
    N = int(n_samples)
    u = _stratified_uniforms(nB, N, dev, generator)
    samples = torch.full((nB, N), -1, dtype=torch.int64, device=dev)
    prev = torch.zeros(nB, dtype=torch.float64, device=dev)
    it = cache if use_cache else ((s, e, None) for s, e in _chunks(ts.n_terms, chunk))
    for s, e, logw in it:
        t = torch.arange(s, e, device=dev, dtype=torch.int64)
        if logw is None:
            logw, _, _, _ = ev(T, ts, t, dtype)
        C = torch.cumsum(torch.exp(logw - logZ[:, None]), dim=1) + prev[:, None]
        j = torch.searchsorted(C.contiguous(), u, right=True)
        inside = (u >= prev[:, None]) & (j < C.shape[1]) & (samples < 0)
        samples = torch.where(inside, t[j.clamp(max=C.shape[1] - 1)], samples)
        prev = C[:, -1]
    # rounding at the very top of the cumulative sum: assign leftovers to the MAP term
    samples = torch.where(samples < 0, best_id[:, None].expand_as(samples), samples)
    out["samples"] = samples
    if want_pip:
        out["pip"], out["spip"] = _mc_pips(ts, samples, M)
    return out

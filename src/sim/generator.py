"""Locus generator (DESIGN v4 §2.2-2.4): studies, effects, realised noncentralities, z/beta/se, reference
LD per mismatch level, rho-hat. Produces eval.io.LocusData + a truth record.

Two stages per locus, so that rejection sampling on the oracle class (§2.4) never pays for noise or LD:
  draw_candidate(ctx, cell, g)  -> Candidate: pairing draw, study samples, variant set, architecture,
                                   beta, own effects lambda, realised mu, oracle class (no noise yet)
  finalize(ctx, cand, g)        -> (LocusData, truth dict): noise with the overlap structure, z/beta/se,
                                   mismatch level LD, L4 corruption, rho-hat

Study models (§2.2):
  factor form (N >= 20k): z = sqrt(N) R_pop beta + G_std' u / sqrt(N_pop), u ~ N(0, I_Npop), G_std = the
      synthetic population standardized by population mean/sd. mu = sqrt(N) R_pop beta.
  individual level (N < 20k): N rows of the synthetic population, standardized IN-SAMPLE (X);
      y = X beta + sigma_e eps with sigma_e² = 1 - var(X beta); z_j = OLS t-statistic of y on x_j
      (+ covariates); mu_j = noise-free t: r_j sqrt(df) / sqrt(1 - r_j²), r_j = corr(x_j, X beta)
      (README D4).
Reported statistics are OLS statistics (decision D6(a)):
  individual level: beta, se of the actual regression of y on the allele dosage (+ intercept and, at L3,
      the 2 PC covariates); z = beta/se is the OLS t-statistic;
  factor form: z_gen = mu + noise is the score-scale statistic sqrt(N) r; beta = z_gen se0,
      se0 = 1/sqrt(2 N f (1-f)); se_ols = se0 sqrt(1 - z_gen²/N); z = beta/se_ols.
  lambda = sqrt(N q), q = per-variant PVE (decision D6(b)).
Overlap (§2.2): two factor traits u2 = rho u1 + sqrt(1-rho²) v; small-N trait nested in a factor study:
  eps2_i = c u1_row(i) + sqrt(1-c²) v_i with c = r_p sqrt(N_pop/N1); same-donor individual-level traits:
  eps2 = r_p eps1 + sqrt(1-r_p²) v. Reachability |r_p| <= 0.9 is asserted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from eval.io import LocusData
from sim import params
from sim.architecture import Architecture, draw_architecture, draw_pve
from sim.mosaic import Mosaic
from sim.oracle import oracle_class
from sim.world import pool_genotypes, source_haplotypes, window_variants
from stats.rho import rho_hat_genomewide, rho_hat_window

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L6", "L6c", "L1swap", "L2swap")
SWAP_LEVELS = ("L1swap", "L2swap")
F32 = torch.float32


def assert_no_tf32():
    """DESIGN §5/§7.1.10: TF32 must be off for every matmul we run."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest", torch.get_float32_matmul_precision()


# ===================================================================================== specs
@dataclass(frozen=True)
class CellSpec:
    cell_id: str
    level: str
    pairing: str                 # P1 / P2 / P3
    overlap: float               # r_p for P1, rho for P2/P3
    p1_mode: str | None = None   # nested / disjoint / None = 50:50
    p3_mode: str | None = None   # eqtl / lab / None = P3_EQTL_SHARE
    both_signs: bool = True      # P3: random rho sign (training/validation); False for test
    ltype_weights: tuple | None = None   # override of the locus-type mixture (e.g. shifted arm)
    arch: tuple = ()             # architecture overrides ((key, value), ...): p_strong, bg_share, m_max


@dataclass
class TraitSpec:
    kind: str     # gwas / mol
    N: int
    form: str     # factor / indiv
    pop: str      # population name (params.POPULATIONS) or "ADMIX"


@dataclass
class PairSpec:
    pairing: str
    sub: str          # nested / disjoint / factor2 / eqtl / lab
    t1: TraitSpec
    t2: TraitSpec
    r_p: float        # per-individual residual correlation among shared donors
    rho: float        # realised null correlation target  N_s r_p / sqrt(N1 N2)
    N_s: int
    same_rows: bool   # individual-level traits measured on the same donors


def level_pops(level: str) -> tuple[str, str, str]:
    """(target population, trait-1 population, trait-2 population) for a mismatch level (DESIGN §2.3)."""
    target = "SWAP" if level in SWAP_LEVELS else "NFE"
    pop1 = {"L6": "FIN", "L6c": "L6c"}.get(level, target)
    pop2 = "ADMIX" if level == "L3" else target
    return target, pop1, pop2


def draw_pair_spec(cell: CellSpec, g: np.random.Generator) -> PairSpec:
    target, pop1, pop2 = level_pops(cell.level)
    ov = float(cell.overlap)
    if cell.level in ("L3", "L6", "L6c") and ov != 0.0:
        raise ValueError(f"{cell.level} is only combined with zero overlap (DESIGN v4 §2.2, §8.6)")
    if cell.pairing == "P1":
        N1 = int(g.choice(params.N1_GRID))
        N2 = int(g.choice(params.N2_GRID_P1))
        mode = cell.p1_mode or ("nested" if g.random() < 0.5 else "disjoint")
        r_p = ov if mode == "nested" else 0.0
        if abs(r_p) > params.MAX_RP:
            raise ValueError(f"unreachable overlap r_p={r_p}")
        N_s = N2 if mode == "nested" else 0
        rho = N_s * r_p / math.sqrt(N1 * N2)
        return PairSpec("P1", mode, TraitSpec("gwas", N1, "factor", pop1), TraitSpec("mol", N2, "indiv", pop2),
                        r_p, rho, N_s, False)
    if cell.pairing == "P2":
        N1, N2 = int(g.choice(params.N1_GRID_P2)), int(g.choice(params.N2_GRID_P2))
        rho = ov
        r_p = rho * math.sqrt(N1 / N2)
        if abs(r_p) > params.MAX_RP:
            raise ValueError(f"P2 rho={rho} unreachable: r_p={r_p:.3f} > {params.MAX_RP}")
        return PairSpec("P2", "factor2", TraitSpec("gwas", N1, "factor", pop1), TraitSpec("mol", N2, "factor", pop2),
                        r_p, rho, N2, False)
    if cell.pairing == "P3":
        mode = cell.p3_mode or ("eqtl" if g.random() < params.P3_EQTL_SHARE else "lab")
        sign = float(g.choice([-1.0, 1.0])) if (cell.both_signs and ov != 0) else 1.0
        rho = sign * abs(ov) if cell.both_signs else ov
        if abs(rho) > params.MAX_RP:
            raise ValueError(f"unreachable overlap rho={rho}")
        if mode == "eqtl":
            N = int(g.choice(params.P3_EQTL_N))
            return PairSpec("P3", "eqtl", TraitSpec("mol", N, "indiv", pop1), TraitSpec("mol", N, "indiv", pop2),
                            rho, rho, N, pop1 == pop2)
        N = params.P3_LAB_N
        return PairSpec("P3", "lab", TraitSpec("gwas", N, "factor", pop1), TraitSpec("gwas", N, "factor", pop2),
                        rho, rho, N, False)
    raise ValueError(cell.pairing)


# ===================================================================================== window context
class PopData:
    """A synthetic (or real) population at the window variants, on the GPU."""

    def __init__(self, name: str, G: torch.Tensor):
        self.name = name
        self.G = G                                       # uint8 [N, p]
        self.N = G.shape[0]
        Gf = G.to(F32)
        self.af = (Gf.mean(0) / 2).double().cpu().numpy()
        self.mean = Gf.mean(0)
        sd = Gf.std(0, unbiased=False)
        self.poly = sd > 0
        self.sd = torch.where(self.poly, sd, torch.ones_like(sd))
        self._std = None
        del Gf

    @property
    def std(self) -> torch.Tensor:
        if self._std is None:
            self._std = (self.G.to(F32) - self.mean) / self.sd
        return self._std

    def drop_std(self):
        self._std = None


class WindowContext:
    """All populations / references of one window, materialised lazily on the GPU."""

    def __init__(self, chrom: str, start: int, end: int, split: str = "", block_ids=(), anchor_block: str = "",
                 anchor: int = -1, device="cuda"):
        self.chrom, self.start, self.end = chrom, int(start), int(end)
        self.split, self.block_ids, self.anchor_block, self.anchor = split, tuple(block_ids), anchor_block, anchor
        self.device = torch.device(device)
        self.v = window_variants(chrom, start, end)
        self.p = len(self.v)
        self._pops: dict[str, PopData] = {}
        self._src: dict[str, np.ndarray] = {}
        self._real: dict[str, PopData] = {}
        self.pc_panel = None                             # set by the caller for L3 (sim.pcs)

    def _source(self, pool: str) -> np.ndarray:
        if pool not in self._src:
            H, v = source_haplotypes(pool, self.chrom, self.start, self.end)
            assert np.array_equal(v["pos"].to_numpy(), self.v["pos"].to_numpy())
            self._src[pool] = H
        return self._src[pool]

    def pop(self, name: str) -> PopData:
        if name not in self._pops:
            m = Mosaic.named(name)
            h = m.haplotypes_from_source(self.chrom, self._source(m.pool), self.v, self.device)
            self._pops[name] = PopData(name, h[0::2] + h[1::2])
            del h
        return self._pops[name]

    def real(self, pool: str) -> PopData:
        """Real genotypes of a pool (references L2-L6: pool B; L2swap: pool A)."""
        if pool not in self._real:
            G, v = pool_genotypes(pool, self.chrom, self.start, self.end)
            assert np.array_equal(v["pos"].to_numpy(), self.v["pos"].to_numpy())
            self._real[pool] = PopData(pool, torch.as_tensor(G.astype(np.uint8), device=self.device))
        return self._real[pool]

    def reference(self, level: str) -> PopData | None:
        if level == "L0":
            return None
        if level == "L1":
            return self.pop("NFE_L1")
        if level == "L1swap":
            return self.pop("SWAP_L1")
        if level == "L2swap":
            return self.real("A")
        return self.real("B")

    def release(self):
        self._pops.clear()
        self._real.clear()
        torch.cuda.empty_cache()


# ===================================================================================== studies
@dataclass
class Study:
    spec: TraitSpec
    pop: PopData | None = None            # factor form: the population
    rows: list = field(default_factory=list)   # indiv: [(PopData, row index array)]
    X: torch.Tensor | None = None         # indiv: in-sample standardized dosages [N, p_win] (generates y)
    Xr: torch.Tensor | None = None        # indiv: X residualised on the covariates (= X without covariates)
    xnorm: torch.Tensor | None = None     # indiv: column norms of Xr
    Q: torch.Tensor | None = None         # indiv: orthonormal covariate basis incl. intercept [N, k]
    af: np.ndarray | None = None          # ALT frequency (in-sample, or population for factor form)
    gsd: torch.Tensor | None = None       # indiv: in-sample SD of the raw ALT dosage (OLS beta scale)

    @property
    def maf(self) -> np.ndarray:
        return np.minimum(self.af, 1 - self.af)

    @property
    def n_cov(self) -> int:
        return 1 if self.Q is None else self.Q.shape[1]


def _pop_for(ctx: WindowContext, name: str) -> PopData:
    return ctx.pop(name)


def make_indiv_study(ctx: WindowContext, spec: TraitSpec, g: np.random.Generator, rows_from=None) -> Study:
    """Individual-level study: N rows of the synthetic population (ADMIX: 85% target + 15% ASW/ACB mosaic)."""
    if rows_from is not None:
        rows = rows_from.rows
    elif spec.pop == "ADMIX":
        n_adm = int(round(params.ADMIX_FRACTION * spec.N))
        A, B = ctx.pop("NFE"), ctx.pop("ADM")
        rows = [(A, np.sort(g.choice(A.N, spec.N - n_adm, replace=False))),
                (B, np.sort(g.choice(B.N, n_adm, replace=False)))]
    else:
        P = _pop_for(ctx, spec.pop)
        rows = [(P, np.sort(g.choice(P.N, spec.N, replace=False)))]
    G = torch.cat([P.G[torch.as_tensor(r, device=ctx.device)] for P, r in rows], 0).to(F32)
    af = (G.mean(0) / 2).double().cpu().numpy()
    X = G - G.mean(0)
    sd = X.std(0, unbiased=False)
    X = X / torch.where(sd > 0, sd, torch.ones_like(sd))
    Q, Xr = None, X
    gsd = sd
    if spec.pop == "ADMIX":
        pcs = admix_pcs(ctx, rows)
        C = torch.cat([torch.ones(spec.N, 1, device=ctx.device, dtype=torch.float64), pcs], 1)
        Q, _ = torch.linalg.qr(C)
        Q = Q.to(F32)
        Xr = X - Q @ (Q.T @ X)
    st = Study(spec, rows=rows, X=X, Xr=Xr, af=af, Q=Q, gsd=gsd)
    st.xnorm = torch.linalg.vector_norm(Xr, dim=0)
    return st


def admix_pcs(ctx: WindowContext, rows) -> torch.Tensor:
    """Top-2 genotype PCs of the study sample from the genome-wide PC panel (sim.pcs, README D6)."""
    if ctx.pc_panel is None:
        from sim.pcs import load_pc_panel
        ctx.pc_panel = load_pc_panel(ctx.device)
    parts = []
    for P, r in rows:
        key = "NFE" if P.name == "NFE" else "ADM"
        parts.append(ctx.pc_panel[key][torch.as_tensor(r, device=ctx.device)])
    M = torch.cat(parts, 0).double()
    M = M - M.mean(0)
    sd = M.std(0)
    M = M / torch.where(sd > 0, sd, torch.ones_like(sd))
    return _top_left_singular(M, 2)


def _top_left_singular(M: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k left singular vectors of M [n, q] via the eigendecomposition of the smaller Gram matrix (fp64).
    Same span as torch.linalg.svd (signs arbitrary; only the span enters the covariate projection), but
    ~50x faster on the GPU than gesvd for n = 20,000 x q = 4,000."""
    n, q = M.shape
    if n <= q:
        ev, U = torch.linalg.eigh(M @ M.T)
        return U[:, -k:].flip(1)
    ev, V = torch.linalg.eigh(M.T @ M)
    V = V[:, -k:].flip(1)
    U = M @ V
    return U / torch.linalg.vector_norm(U, dim=0)


def make_factor_study(ctx: WindowContext, spec: TraitSpec) -> Study:
    if spec.pop == "ADMIX":
        P = admix_factor_pop(ctx)
    else:
        P = _pop_for(ctx, spec.pop)
    return Study(spec, pop=P, af=P.af)


_PC_CACHE: dict = {}


def admix_factor_pop(ctx: WindowContext) -> PopData:
    """L3 factor-form admixed population: 85% target-mosaic rows + 15% ASW/ACB-mosaic rows, PC-residualised."""
    if "ADMIX_FACTOR" not in ctx._pops:
        A, B = ctx.pop("NFE"), ctx.pop("ADM")
        n_b = B.N
        n_a = int(round(n_b * (1 - params.ADMIX_FRACTION) / params.ADMIX_FRACTION))
        rows = [(A, np.arange(n_a)), (B, np.arange(n_b))]
        P = PopData("ADMIX_FACTOR", torch.cat([A.G[:n_a], B.G], 0))
        # the rows (and the genome-wide PC panel) are the same for every window: compute the PCs once
        key = ("ADMIX_FACTOR", n_a, n_b, str(ctx.device))
        if key not in _PC_CACHE:
            _PC_CACHE[key] = admix_pcs(ctx, rows)
        pcs = _PC_CACHE[key]
        C = torch.cat([torch.ones(P.N, 1, device=ctx.device, dtype=torch.float64), pcs], 1)
        Q, _ = torch.linalg.qr(C)
        Q = Q.to(F32)
        Xs = P.std
        Xs = Xs - Q @ (Q.T @ Xs)
        sd = Xs.std(0, unbiased=False)
        P._std = Xs / torch.where(sd > 0, sd, torch.ones_like(sd))
        ctx._pops["ADMIX_FACTOR"] = P
    return ctx._pops["ADMIX_FACTOR"]


def make_studies(ctx: WindowContext, ps: PairSpec, g: np.random.Generator) -> tuple[Study, Study]:
    s1 = make_factor_study(ctx, ps.t1) if ps.t1.form == "factor" else make_indiv_study(ctx, ps.t1, g)
    if ps.t2.form == "factor":
        s2 = make_factor_study(ctx, ps.t2)
    else:
        s2 = make_indiv_study(ctx, ps.t2, g, rows_from=s1 if ps.same_rows else None)
    return s1, s2


# ===================================================================================== candidate
@dataclass
class Candidate:
    cell: CellSpec
    ps: PairSpec
    s1: Study
    s2: Study
    keep: np.ndarray                 # analysis variants (indices into the window) before L4 dropping
    arch: Architecture
    beta1: torch.Tensor              # [p_win] per-trait standardized effects (trait population scale)
    beta2: torch.Tensor
    lam1: np.ndarray                 # own effects at arch.c1 (window indices)
    lam2: np.ndarray
    mu1: torch.Tensor                # realised noncentralities over the window [p_win]
    mu2: torch.Tensor
    oracle: object
    r2_true_cd: float
    pve1: float
    pve2: float
    seed_key: tuple = ()


def _target_r2_row(ctx: WindowContext, target: PopData, keep_t: torch.Tensor, j_win: int) -> np.ndarray:
    Z = target.std
    r = (Z[:, keep_t].T @ Z[:, j_win]) / target.N
    return (r * r).double().cpu().numpy()


def _r2_pair(target: PopData, a: int, b: int) -> float:
    Z = target.std
    return float(((Z[:, a] @ Z[:, b]) / target.N) ** 2)


def _beta_from_specs(ctx, st: Study, specs: dict, target: PopData, g, bg: bool, keep_t) -> torch.Tensor:
    p = ctx.p
    beta = torch.zeros(p, dtype=F32, device=ctx.device)
    N = st.spec.N
    for j, spec in specs.items():
        if spec[0] == "pve":
            b = spec[2] * math.sqrt(spec[1])
            if st.spec.pop in ("FIN", "L6c"):
                # allelic-scale effect converted by population allele frequency (DESIGN §2.3, L6)
                ft, fp = target.af[j], ctx.pop(st.spec.pop).af[j]
                b = b * math.sqrt(fp * (1 - fp)) / math.sqrt(ft * (1 - ft)) if fp * (1 - fp) > 0 else 0.0
        else:
            b = spec[1] / math.sqrt(N)
        beta[j] = b
    if bg:
        k = len(keep_t)
        bgv = g.normal(0.0, math.sqrt(params.BG_SUM_LAMBDA2 / (N * k)), size=k)
        beta[keep_t] += torch.as_tensor(bgv, dtype=F32, device=ctx.device)
    return beta


def _genetic_signal(st: Study, beta: torch.Tensor) -> torch.Tensor:
    """Per-individual genetic value s = X beta (indiv) or G_std beta (factor population)."""
    if st.spec.form == "indiv":
        return st.X @ beta
    return st.pop.std @ beta


def _pve(st: Study, s: torch.Tensor) -> float:
    return float((s.double() ** 2).mean() - s.double().mean() ** 2)


def _mu(st: Study, beta: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    N = st.spec.N
    if st.spec.form == "factor":
        P = st.pop
        return math.sqrt(N) * (P.std.T @ s) / P.N
    # noise-free t-statistic: r_j = corr(x_j, s) * sd(s) / sd(y), sd(y) = 1 by construction (sigma_e² = 1 - var s)
    sc = s - s.mean() if st.Q is None else s - st.Q @ (st.Q.T @ s)
    var_s = float((s.double() ** 2).mean() - s.double().mean() ** 2)
    sig_e2 = 1.0 - var_s
    df = N - st.n_cov - 1
    ytot = math.sqrt(float((sc.double() ** 2).sum()) + sig_e2 * (N - st.n_cov))
    r = (st.Xr.T @ sc) / (st.xnorm.clamp_min(1e-12) * ytot)
    r = r.clamp(-0.999999, 0.999999)
    return r * math.sqrt(df) / torch.sqrt(1 - r * r)


def variant_mask(ctx: WindowContext, s1: Study, s2: Study, level: str) -> np.ndarray:
    m = (s1.maf >= params.MIN_MAF_STUDY) & (s2.maf >= params.MIN_MAF_STUDY)
    ref = ctx.reference(level)
    if ref is not None:
        m &= ref.poly.cpu().numpy()
    return np.where(m)[0]


def draw_candidate(ctx: WindowContext, cell: CellSpec, g: np.random.Generator, seed_key=()) -> Candidate | None:
    ps = draw_pair_spec(cell, g)
    s1, s2 = make_studies(ctx, ps, g)
    keep = variant_mask(ctx, s1, s2, cell.level)
    if len(keep) < 50:
        return None
    target_name = level_pops(cell.level)[0]
    target = ctx.pop(target_name)
    keep_t = torch.as_tensor(keep, device=ctx.device)
    maf_t = np.minimum(target.af[keep], 1 - target.af[keep])
    ltype = None
    if cell.ltype_weights is not None:
        types, w = zip(*cell.ltype_weights)
        ltype = types[g.choice(len(types), p=np.asarray(w) / np.sum(w))]
    arch = draw_architecture(g, maf_t, lambda j: _target_r2_row(ctx, target, keep_t, int(keep[j])),
                             ps.t1.kind, ps.t2.kind, ltype=ltype, **dict(cell.arch))
    # map architecture indices (into keep) to window indices
    arch.c1 = [int(keep[j]) for j in arch.c1]
    arch.c2 = [int(keep[j]) for j in arch.c2]
    arch.eff1 = {int(keep[j]): s for j, s in arch.eff1.items()}
    arch.eff2 = {int(keep[j]): s for j, s in arch.eff2.items()}
    for _ in range(50):
        beta1 = _beta_from_specs(ctx, s1, arch.eff1, target, g, arch.bg, keep_t)
        beta2 = _beta_from_specs(ctx, s2, arch.eff2, target, g, arch.bg, keep_t)
        sig1, sig2 = _genetic_signal(s1, beta1), _genetic_signal(s2, beta2)
        pve1, pve2 = _pve(s1, sig1), _pve(s2, sig2)
        if pve1 <= params.MAX_TOTAL_PVE and pve2 <= params.MAX_TOTAL_PVE:
            break
        _redraw_pves(arch, g, ps)       # DESIGN v4 §1: redraw architectures exceeding the PVE cap
    else:
        return None
    mu1, mu2 = _mu(s1, beta1, sig1), _mu(s2, beta2, sig2)
    lam1 = np.array([math.sqrt(ps.t1.N) * float(beta1[j]) for j in arch.c1])
    lam2 = np.array([math.sqrt(ps.t2.N) * float(beta2[j]) for j in arch.c2])
    m1 = mu1[torch.as_tensor(arch.c1, device=ctx.device, dtype=torch.long)].double().cpu().numpy() if arch.c1 else np.array([])
    m2 = mu2[torch.as_tensor(arch.c2, device=ctx.device, dtype=torch.long)].double().cpu().numpy() if arch.c2 else np.array([])
    orc = oracle_class(arch.c1, lam1, m1, arch.c2, lam2, m2, lambda a, b: _r2_pair(target, a, b),
                       ps.t1.kind, ps.t2.kind)
    return Candidate(cell, ps, s1, s2, keep, arch, beta1, beta2, lam1, lam2, mu1, mu2, orc,
                     orc.r2, pve1, pve2, seed_key)


def _redraw_pves(arch: Architecture, g, ps: PairSpec):
    for t, eff in ((1, arch.eff1), (2, arch.eff2)):
        kind = ps.t1.kind if t == 1 else ps.t2.kind
        strong = arch.strong1 if t == 1 else arch.strong2
        for j, s in list(eff.items()):
            if s[0] == "pve":
                eff[j] = ("pve", draw_pve(g, kind, strong), s[2])


# ===================================================================================== finalize
def _std_ref(P: PopData, cols: torch.Tensor) -> torch.Tensor:
    X = P.G[:, cols].to(F32)
    X = X - X.mean(0)
    sd = X.std(0, unbiased=False)
    return X / torch.where(sd > 0, sd, torch.ones_like(sd))


def _ld_payload(X: torch.Tensor | None, R: torch.Tensor | None):
    """Return (ld_type, R or None, X or None, n_ref); factor if n < p else dense."""
    if R is not None:
        return "dense", R.cpu().numpy().astype(np.float32), None, None
    n, p = X.shape
    if n < p:
        return "factor", None, X.cpu().numpy().astype(np.float32), int(n)
    return "dense", (X.T @ X / n).cpu().numpy().astype(np.float32), None, int(n)


def rho_hat_for(ps: PairSpec, z1: np.ndarray, z2: np.ndarray, src, g, level: str = "") -> dict:
    """rho-hat (README D14): per-window (P1, P3 eQTL x eQTL) on the locus' z with the LD given to methods
    (`src`: LDSource or dense tensor), or genome-wide (P2, P3 lab x lab; 20,000 simulated independent SNPs with
    the study's true rho). Signal residualisation only with accurate LD (L0 in-sample, or n_ref >= 1000)."""
    if ps.pairing == "P2" or (ps.pairing == "P3" and ps.sub == "lab"):
        return rho_hat_genomewide(ps.rho, g, params.RHO_GW_NSNP, params.RHO_GW_POLY, params.RHO_GW_POLY_SD,
                                  params.RHO_ZMAX, params.RHO_JK_BLOCKS)
    Rd = (src.full() if isinstance(src, LDSource) else src).float().cpu().numpy()
    n_ref = src.X.shape[0] if isinstance(src, LDSource) else 10 ** 9
    level = level or (src.level if isinstance(src, LDSource) else "")
    resid = level == "L0" or n_ref >= params.RHO_RESID_MIN_NREF
    order = g.permutation(len(z1))
    return rho_hat_window(z1, z2, Rd, params.RHO_PRUNE_R2, params.RHO_ZMAX, order, z_signal=params.RHO_Z_SIGNAL,
                          mask_r2=params.RHO_MASK_R2, n_blocks=params.RHO_JK_BLOCKS, residualize_signal=resid)


@dataclass
class LDSource:
    """LD given to methods for one locus (DESIGN §2.3), as a standardized factor X [n, p_locus]; R = X'X/n."""
    X: torch.Tensor
    n_ref_attr: int               # n_ref written to the locus (population / study / reference size)
    maf_ref: np.ndarray
    key: str | None               # identity for LD sharing (None = locus-specific)
    level: str = ""               # mismatch level (used by rho_hat_for: residualise with in-sample LD at L0)

    def rows(self, idx) -> torch.Tensor:
        idx = torch.as_tensor(np.asarray(idx), device=self.X.device, dtype=torch.long)
        return (self.X[:, idx].T @ self.X) / self.X.shape[0]

    def block(self, idx) -> torch.Tensor:
        idx = torch.as_tensor(np.asarray(idx), device=self.X.device, dtype=torch.long)
        Xi = self.X[:, idx]
        return (Xi.T @ Xi) / self.X.shape[0]

    def full(self) -> torch.Tensor:
        return (self.X.T @ self.X) / self.X.shape[0]


def ld_source(ctx: WindowContext, c: Candidate, keep: np.ndarray) -> LDSource:
    """L0: in-sample LD of trait 1's study (population LD for factor form); L1-L6: the level's reference."""
    kt = torch.as_tensor(keep, device=ctx.device)
    s1, level = c.s1, c.cell.level
    if level == "L0":
        f1 = s1.af[keep]
        if s1.spec.form == "factor":
            return LDSource(s1.pop.std[:, kt], s1.pop.N, np.minimum(f1, 1 - f1), _ld_key(ctx, "L0", s1.pop.name, keep),
                            "L0")
        return LDSource(_restd(s1.Xr[:, kt]), s1.spec.N, np.minimum(f1, 1 - f1), None, "L0")
    ref = ctx.reference(level)
    return LDSource(_std_ref(ref, kt), ref.N, np.minimum(ref.af[keep], 1 - ref.af[keep]),
                    _ld_key(ctx, level, ref.name, keep), level)


def observed_stats(ctx: WindowContext, c: Candidate, g: np.random.Generator) -> dict:
    """Noise draw with the overlap structure, OLS statistics, L4 corruption. Returns keep (window indices of
    the locus variants), z/beta/se per trait (numpy, locus order), flipped indices/traits."""
    ps, s1, s2 = c.ps, c.s1, c.s2
    st1 = noisy_stats(s1, c.beta1, c.mu1, g)
    st2 = noisy_stats(s2, c.beta2, c.mu2, g, partner=(ps, s1, st1["eps"], st1["u"]))
    keep = c.keep.copy()
    flipped = np.empty(0, np.int64)
    flip_trait = np.empty(0, np.int64)
    if c.cell.level == "L4":
        n_drop = int(round(params.L4_MISSING * len(keep)))
        drop = np.sort(g.choice(len(keep), n_drop, replace=False))
        keep = np.delete(keep, drop)
        n_flip = int(round(params.L4_FLIP * len(keep)))
        flipped = np.sort(g.choice(len(keep), n_flip, replace=False))
        flip_trait = g.integers(1, 3, size=n_flip)
    kt = torch.as_tensor(keep, device=ctx.device)
    b1, se1 = st1["beta"][kt].cpu().numpy(), st1["se"][kt].cpu().numpy()
    b2, se2 = st2["beta"][kt].cpu().numpy(), st2["se"][kt].cpu().numpy()
    for i, t in zip(flipped, flip_trait):     # L4: allele flipped in one trait (beta and z change sign)
        if t == 1:
            b1[i] = -b1[i]
        else:
            b2[i] = -b2[i]
    return {"keep": keep, "b1": b1, "se1": se1, "b2": b2, "se2": se2, "z1": b1 / se1, "z2": b2 / se2,
            "flipped": flipped, "flip_trait": flip_trait}


def finalize(ctx: WindowContext, c: Candidate, g: np.random.Generator, locus_id: str) -> tuple[LocusData, dict, dict]:
    """Returns (LocusData, truth record, extra) where extra may hold 'ld2' = trait 2's in-sample LD at L0."""
    ps, s1, s2, dev = c.ps, c.s1, c.s2, ctx.device
    ob = observed_stats(ctx, c, g)
    keep, b1, se1, b2, se2, z1, z2 = (ob[k] for k in ("keep", "b1", "se1", "b2", "se2", "z1", "z2"))
    flipped, flip_trait = ob["flipped"], ob["flip_trait"]
    kt = torch.as_tensor(keep, device=dev)
    f1, f2 = s1.af[keep], s2.af[keep]
    # ---------------------------------------------------------------- LD given to methods
    level = c.cell.level
    extra = {}
    src = ld_source(ctx, c, keep)
    n_ref, maf_ref, ld_key = src.n_ref_attr, src.maf_ref, src.key
    n_x, p_loc = src.X.shape
    if n_x < p_loc:
        ld_type, R, X = "factor", None, src.X.cpu().numpy().astype(np.float32)
        n_ref = n_x
    else:
        ld_type, R, X = "dense", src.full().cpu().numpy().astype(np.float32), None
    if level == "L0" and s2.spec.form == "indiv" and not ps.same_rows:
        lt2, R2, X2, n2r = _ld_payload(_restd(s2.Xr[:, kt]), None)
        extra["ld2"] = (lt2, R2, X2, s2.spec.N)
    # ---------------------------------------------------------------- rho-hat
    rh = rho_hat_for(ps, z1, z2, src, g, level)
    # ---------------------------------------------------------------- assemble
    v = ctx.v.iloc[keep]
    win_to_loc = {int(w): i for i, w in enumerate(keep)}
    loc = LocusData(
        locus_id=locus_id, variant_id=v["variant_id"].to_numpy(object), pos=v["pos"].to_numpy(np.int64),
        z1=z1, z2=z2, beta1=b1, se1=se1, beta2=b2, se2=se2,
        n1=np.full(len(keep), float(ps.t1.N)), n2=np.full(len(keep), float(ps.t2.N)),
        maf1=np.minimum(f1, 1 - f1), maf2=np.minimum(f2, 1 - f2), maf_ref=maf_ref,
        ld_type=ld_type, n_ref=int(n_ref), R=R, X_ref=X,
        rho_hat=float(rh["rho_hat"]), rho_se=float(rh["rho_se"]), split=ctx.split,
        block_ids=tuple(ctx.block_ids), mismatch_level=level, pairing_type=ps.pairing, cell_id=c.cell.cell_id,
        window_chr=ctx.chrom, window_start=ctx.start, window_end=ctx.end,
        extra_attrs={"trait1_type": ps.t1.kind, "trait2_type": ps.t2.kind}, ld_key=ld_key,
    )
    o = c.oracle
    idx1 = [win_to_loc.get(j, -1) for j in c.arch.c1]
    idx2 = [win_to_loc.get(j, -1) for j in c.arch.c2]
    m1 = c.mu1[torch.as_tensor(c.arch.c1, device=dev, dtype=torch.long)].double().cpu().numpy().tolist() if c.arch.c1 else []
    m2 = c.mu2[torch.as_tensor(c.arch.c2, device=dev, dtype=torch.long)].double().cpu().numpy().tolist() if c.arch.c2 else []
    truth = {
        "locus_id": locus_id, "oracle_class": o.cls, "mu1": o.mu1, "mu2": o.mu2, "r2": o.r2, "d": o.d,
        "rho": ps.rho, "block_id": ctx.anchor_block, "cell_id": c.cell.cell_id, "split": ctx.split,
        "mismatch_level": level, "pairing_type": ps.pairing, "overlap": c.cell.overlap, "control": "",
        "causal1": idx1, "causal2": idx2, "effect1": c.lam1.tolist(), "effect2": c.lam2.tolist(),
        "mu1_all": m1, "mu2_all": m2,
        "causal1_vid": ctx.v["variant_id"].to_numpy()[c.arch.c1].tolist(),
        "causal2_vid": ctx.v["variant_id"].to_numpy()[c.arch.c2].tolist(),
        "cd1": win_to_loc.get(o.cd1, -1) if o.cd1 is not None else -1,
        "cd2": win_to_loc.get(o.cd2, -1) if o.cd2 is not None else -1,
        "n_powered1": o.n_powered1, "n_powered2": o.n_powered2,
        "locus_type": c.arch.ltype, "target_r2_bin": c.arch.target_r2_bin, "background": c.arch.bg,
        "strong1": c.arch.strong1, "strong2": c.arch.strong2, "pve1": c.pve1, "pve2": c.pve2,
        "pairing_sub": ps.sub, "N1": ps.t1.N, "N2": ps.t2.N, "N_s": ps.N_s, "r_p": ps.r_p,
        "trait1_type": ps.t1.kind, "trait2_type": ps.t2.kind, "pop1": ps.t1.pop, "pop2": ps.t2.pop,
        "form1": ps.t1.form, "form2": ps.t2.form,
        "rho_hat": float(rh["rho_hat"]), "rho_se": float(rh["rho_se"]), "rho_mode": rh["mode"],
        "rho_n_used": int(rh["n_used"]), "p": int(len(keep)), "p_study": int(len(c.keep)), "p_window": int(ctx.p),
        "flipped": [int(i) for i in flipped], "flip_trait": [int(t) for t in flip_trait],
        "lead_abs_z1": float(np.max(np.abs(z1))), "lead_abs_z2": float(np.max(np.abs(z2))),
        "anchor": int(ctx.anchor), "window_chr": ctx.chrom, "window_start": ctx.start, "window_end": ctx.end,
        "seed_key": repr(c.seed_key),
    }
    return loc, truth, extra


def _ld_key(ctx: WindowContext, level: str, source: str, keep: np.ndarray) -> str:
    """Identity of a deterministic LD payload (same window, LD source and variant set -> same matrix), so a
    locus-set writer can store it once (eval.io ld_key hard links)."""
    import hashlib
    h = hashlib.sha1(f"{ctx.chrom}:{ctx.start}-{ctx.end}|{level}|{source}|".encode())
    h.update(np.ascontiguousarray(keep, dtype=np.int64).tobytes())
    return h.hexdigest()[:20]


def _restd(X: torch.Tensor) -> torch.Tensor:
    X = X - X.mean(0)
    sd = X.std(0, unbiased=False)
    return X / torch.where(sd > 0, sd, torch.ones_like(sd))


def noisy_stats(st: Study, beta, mu, g, partner=None) -> dict:
    """One noise draw with the overlap structure; returns OLS statistics over the window (decision D6(a)):
    dict(z, beta, se) as torch tensors [p_win] (z = beta/se), plus eps (indiv) / u (factor) for a partner."""
    dev = beta.device
    N = st.spec.N
    if st.spec.form == "factor":
        P = st.pop
        u = g.standard_normal(P.N)
        if partner is not None:
            ps, s1, eps1, u1 = partner
            if ps.rho != 0.0:
                assert s1.spec.form == "factor" and s1.pop is P, "factor overlap needs one population"
                u = ps.rho * u1 + math.sqrt(1 - ps.rho ** 2) * u
        ut = torch.as_tensor(u, dtype=F32, device=dev)
        zg = (mu + (P.std.T @ ut) / math.sqrt(P.N)).double()          # score scale sqrt(N) r
        f = torch.as_tensor(st.af, dtype=torch.float64, device=dev)
        se0 = 1.0 / torch.sqrt((2.0 * N * f * (1 - f)).clamp_min(1e-300))
        shrink = 1.0 - zg * zg / N
        assert float(shrink.min()) > 0.01, "factor-form z too large for its N"
        se = se0 * torch.sqrt(shrink)
        b = zg * se0
        return {"z": b / se, "beta": b, "se": se, "eps": None, "u": u}
    # individual level
    s = st.X @ beta
    var_s = _pve(st, s)
    sig_e = math.sqrt(1.0 - var_s)
    eps = g.standard_normal(N)
    if partner is not None:
        ps, s1, eps1, u1 = partner
        if ps.r_p != 0.0 and ps.N_s > 0:
            if s1.spec.form == "indiv":
                assert ps.same_rows
                eps = ps.r_p * eps1 + math.sqrt(1 - ps.r_p ** 2) * eps
            else:
                # nested in a factor-form study: rows of the shared population carry its noise
                c = ps.r_p * math.sqrt(s1.pop.N / s1.spec.N)
                assert abs(c) <= 1.0, c
                assert len(st.rows) == 1 and st.rows[0][0] is s1.pop, "nested rows must come from trait 1's population"
                eps = c * u1[st.rows[0][1]] + math.sqrt(1 - c * c) * eps
    y = s + sig_e * torch.as_tensor(eps, dtype=F32, device=dev)
    yc = y - y.mean() if st.Q is None else y - st.Q @ (st.Q.T @ y)
    ynorm = torch.linalg.vector_norm(yc.double())
    xn = st.xnorm.double().clamp_min(1e-12)
    r = ((st.Xr.T @ yc).double() / (xn * ynorm)).clamp(-0.999999, 0.999999)
    df = N - st.n_cov - 1
    gsd = st.gsd.double().clamp_min(1e-12)
    b = r * ynorm / (gsd * xn)                                  # OLS slope on the ALT dosage
    se = ynorm * torch.sqrt((1 - r * r) / df) / (gsd * xn)      # OLS SE (residual variance on df)
    return {"z": b / se, "beta": b, "se": se, "eps": eps, "u": None}


def _noisy_z(st: Study, beta, mu, g, partner=None):
    """Return (OLS z over the window, eps (indiv) or None, u (factor) or None)."""
    d = noisy_stats(st, beta, mu, g, partner)
    return d["z"].float(), d["eps"], d["u"]

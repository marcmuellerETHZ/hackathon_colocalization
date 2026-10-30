"""Batched GPU screen + featurisation for NeuralColoc (DESIGN v4 §5 C "Input").

Screen: the per-trait-union screen of Family B (`famB.screen.screen`, collapse=False): lists m1, m2 (marginal |z_t|)
and c1, c2 (lead-removed standardized conditional |z_t|), quotas (M/2 - M/8, M/2 - M/8, M/8, M/8), visited
round-robin m1, m2, c1, c2, each list taking its next not-yet-selected variant. The duplicate collapse of B is not
applied (the network sees near-duplicates as separate tokens). If p <= M every variant is selected (source 4).
Identity with famB.screen(collapse=False) is tested (tests/famC/test_features.py).

Allele-flip invariance (sign-equivariant treatment of z): every screened variant j is re-coded by
s_j = sign(z_{t*,j}), t* = the trait with the larger |z_j|; token features use s_j z_tj and the LD bias uses
s_i s_j R_ij. Both are exactly invariant to flipping the coding allele of any variant (tested). The all-variant
stream only uses allele-invariant quantities (z1², z2², z1 z2, conditional analogues, r²).
Global trait-sign flips (z_t -> -z_t for all variants, rho -> -rho) are handled by augmentation in training.

LD access objects:
  FactorLD(X [Wb,n,P], loci_per_window L): loci b = w*L + l share window w's standardized genotypes; R = X'X/n.
  DenseLD(R [B,P,P]) for small batches of loci with their own dense LD (padding rows/cols zero).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

ETA = 1e-3
N_TOK_FEAT = 22
N_GLOB_FEAT = 32
N_ALL_FEAT = 10      # coefficient-net inputs of the all-variant stream
N_QUAD = 6           # quadratic energy basis: z1², z2², z1 z2, zc1², zc2², zc1 zc2 (all / 2)
N_EIG = 12


class FactorLD:
    def __init__(self, X: torch.Tensor, L: int, n: int | None = None):
        self.X = X                    # [Wb, n, P] standardized, zero padded
        self.L = L
        self.n = float(n if n is not None else X.shape[1])

    def rows(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [B, k] -> R[idx, :] [B, k, P]."""
        Wb, n, P = self.X.shape
        B, k = idx.shape
        Xi = torch.gather(self.X, 2, idx.reshape(Wb, 1, self.L * k).expand(Wb, n, self.L * k))  # [Wb, n, L*k]
        out = torch.bmm(Xi.transpose(1, 2), self.X) / self.n                                 # [Wb, L*k, P]
        return out.reshape(B, k, P)

    def sub(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [B, M] -> R[idx][:, idx] [B, M, M]."""
        Wb, n, P = self.X.shape
        B, M = idx.shape
        Xi = torch.gather(self.X, 2, idx.reshape(Wb, 1, self.L * M).expand(Wb, n, self.L * M))
        Xi = Xi.reshape(Wb, n, self.L, M).permute(0, 2, 3, 1).reshape(B, M, n)
        return torch.bmm(Xi, Xi.transpose(1, 2)) / self.n


class DenseLD:
    def __init__(self, R: torch.Tensor):
        self.R = R                    # [B, P, P]

    def rows(self, idx):
        B, k = idx.shape
        P = self.R.shape[-1]
        return torch.gather(self.R, 1, idx[:, :, None].expand(B, k, P))

    def sub(self, idx):
        B, M = idx.shape
        r = self.rows(idx)            # [B, M, P]
        return torch.gather(r, 2, idx[:, None, :].expand(B, M, M))


class StackedLD:
    """Per-locus LD objects (each B=1), concatenated."""

    def __init__(self, lds):
        self.lds = lds

    def rows(self, idx):
        P = max(ld_P(l) for l in self.lds)
        out = []
        for b, l in enumerate(self.lds):
            r = l.rows(idx[b:b + 1])
            out.append(torch.nn.functional.pad(r, (0, P - r.shape[-1])))
        return torch.cat(out, 0)

    def sub(self, idx):
        return torch.cat([l.sub(idx[b:b + 1]) for b, l in enumerate(self.lds)], 0)


def ld_P(l):
    if isinstance(l, FactorLD):
        return l.X.shape[2]
    return l.R.shape[-1]


def conditional_z(z: torch.Tensor, r_lead: torch.Tensor, lead: torch.Tensor, eta: float = ETA) -> torch.Tensor:
    """famB.screen.conditional_z, batched. z, r_lead [B,P] float64; lead [B]."""
    B, P = z.shape
    ar = torch.arange(B, device=z.device)
    rt = (1.0 - eta) * r_lead
    rt[ar, lead] = 1.0
    var = (1.0 - rt * rt).clamp_min(1e-12)
    zc = (z - rt * z[ar, lead][:, None]) / torch.sqrt(var)
    zc[ar, lead] = 0.0
    return zc


@dataclass
class ScreenOut:
    sel: torch.Tensor      # [B, M] variant indices (padding: 0)
    valid: torch.Tensor    # [B, M] bool
    source: torch.Tensor   # [B, M] 0 m1, 1 m2, 2 c1, 3 c2, 4 all, -1 padding
    lead1: torch.Tensor
    lead2: torch.Tensor
    rlead: torch.Tensor    # [B, 2, P] LD rows of the leads (inference LD)
    zc1: torch.Tensor      # [B, P]
    zc2: torch.Tensor


@torch.no_grad()
def screen_gpu(z1, z2, mask, ld, M: int = 128, k_cond: int | None = None, eta: float = ETA) -> ScreenOut:
    B, P = z1.shape
    dev = z1.device
    neg = torch.tensor(-math.inf, dtype=z1.dtype, device=dev)
    a1 = torch.where(mask, z1.abs(), neg)
    a2 = torch.where(mask, z2.abs(), neg)
    lead1, lead2 = a1.argmax(-1), a2.argmax(-1)
    rlead = ld.rows(torch.stack([lead1, lead2], 1)).to(z1.dtype)      # [B, 2, P]
    zc1 = conditional_z(z1, rlead[:, 0].clone(), lead1, eta)
    zc2 = conditional_z(z2, rlead[:, 1].clone(), lead2, eta)
    zc1 = torch.where(mask, zc1, torch.zeros_like(zc1))
    zc2 = torch.where(mask, zc2, torch.zeros_like(zc2))
    scores = torch.stack([a1, a2, torch.where(mask, zc1.abs(), neg), torch.where(mask, zc2.abs(), neg)], 0)
    if k_cond is None:
        k_cond = max(1, M // 8)
    half = M // 2
    quota = [half - k_cond, M - half - k_cond, k_cond, k_cond]
    sel, src = _merge_lists(scores, mask, quota, M)
    pw = mask.sum(-1)
    small = pw <= M
    valid = torch.ones(B, M, dtype=torch.bool, device=dev)
    if small.any():
        idx = torch.arange(M, device=dev)[None, :].expand(int(small.sum()), M)
        v = idx < pw[small][:, None]
        sel[small] = torch.where(v, idx, torch.zeros_like(idx))
        src[small] = torch.where(v, torch.full_like(idx, 4), torch.full_like(idx, -1))
        valid[small] = v
    return ScreenOut(sel, valid, src, lead1, lead2, rlead, zc1, zc2)


def _merge_lists(scores: torch.Tensor, mask: torch.Tensor, quota: list[int], M: int):
    """famB round-robin merge (lists visited in order each round, a list takes its next not-yet-selected
    variant until its quota is used). Lists are sorted on the GPU (stable, descending, so ties go to the lower
    index as in famB's stable argsort); the merge runs vectorised over loci in numpy on the top-2M prefixes
    (a list can skip at most M - 1 already-selected variants)."""
    import numpy as np
    L, B, P = scores.shape
    depth = min(P, 2 * M)
    order = torch.sort(scores, dim=-1, descending=True, stable=True).indices[..., :depth].cpu().numpy()
    covered = ~mask.cpu().numpy()
    sel = np.zeros((B, M), dtype=np.int64)
    src = np.full((B, M), -1, dtype=np.int64)
    ptr = np.zeros((L, B), dtype=np.int64)
    ar = np.arange(B)
    pos = 0
    for r in range(max(quota)):
        for li in range(L):
            if r >= quota[li]:
                continue
            while True:
                cand = order[li, ar, np.minimum(ptr[li], depth - 1)]
                skip = covered[ar, cand] & (ptr[li] < depth - 1)
                if not skip.any():
                    break
                ptr[li] += skip
            j = order[li, ar, np.minimum(ptr[li], depth - 1)]
            ptr[li] += 1
            sel[:, pos] = j
            src[:, pos] = li
            covered[ar, j] = True
            pos += 1
    assert pos == M
    dev = scores.device
    return torch.as_tensor(sel, device=dev), torch.as_tensor(src, device=dev)


def _asinh2(x):
    return torch.asinh(x / 2.0)


@torch.no_grad()
def featurize(b: dict, ld, M: int = 128, use_ld: bool = True, flip_traits: torch.Tensor | None = None) -> dict:
    """Model inputs from a batch dict = build(extract(b, ld, M, use_ld), flip_traits). See extract/build."""
    return build(extract(b, ld, M, use_ld), flip_traits)


@torch.no_grad()
def extract(b: dict, ld, M: int = 128, use_ld: bool = True) -> dict:
    """Stage 1 (needs LD): screen and raw, augmentation-free quantities of a batch.

    b: z1, z2, se1, se2, n1, n2, maf1, maf2, maf_ref, mask [B,P]; n_ref (int or [B]); rho_hat, rho_se, type1,
    type2 [B] (trait type 0 = GWAS-like, 1 = molecular). use_ld = False is the no-LD ablation (marginal-only
    screen; no conditional z, LD rows, LD bias or eigen-spectrum).
    The record is invariant to global trait-sign flips except through the signs of z (handled in build).
    The true rho is never an input (DESIGN §2.4; review p2_review_sim note 11): only rho_hat / rho_se are read, and
    a batch that carries a truth field `rho` (as sim.train_api batches do) is refused outright."""
    assert "rho" not in b, "true rho must never reach the network: pass rho_hat / rho_se only (DESIGN §2.4)"
    z1 = b["z1"].double()
    z2 = b["z2"].double()
    mask = b["mask"]
    B, P = z1.shape
    dev = z1.device
    sc = screen_gpu(z1, z2, mask, ld, M=M) if use_ld else _screen_nold(z1, z2, mask, M)
    sel, valid = sc.sel, sc.valid
    g = lambda x: torch.gather(x, 1, sel)
    ar = torch.arange(B, device=dev)
    lse1, lse2 = torch.log(b["se1"].double()), torch.log(b["se2"].double())
    ln1, ln2 = torch.log10(b["n1"].double()), torch.log10(b["n2"].double())
    n_ref = b["n_ref"]
    if not torch.is_tensor(n_ref):
        n_ref = torch.full((B,), float(n_ref), dtype=torch.float64, device=dev)
    zc1 = sc.zc1 if use_ld else torch.zeros_like(z1)
    zc2 = sc.zc2 if use_ld else torch.zeros_like(z2)
    rl = sc.rlead.double() if use_ld else torch.zeros(B, 2, P, dtype=torch.float64, device=dev)
    maf1, maf2, mafr = b["maf1"].double(), b["maf2"].double(), b["maf_ref"].double()
    tok_raw = torch.stack([g(z1), g(z2), g(zc1), g(zc2), g(rl[:, 0]), g(rl[:, 1]), g(lse1), g(lse2), g(maf1),
                           g(maf2), g(mafr), (sel == sc.lead1[:, None]).double(), (sel == sc.lead2[:, None]).double()],
                          -1) * valid[..., None]
    all_raw = torch.stack([z1, z2, zc1, zc2, lse1, lse2, ln1, ln2, maf1, maf2, mafr, rl[:, 0] ** 2, rl[:, 1] ** 2],
                          -1) * mask[..., None]
    if use_ld:
        Rmm = ld.sub(sel).double() * valid[:, :, None] * valid[:, None, :]
        ev = torch.linalg.eigvalsh(Rmm.float() + 1e-6 * torch.eye(M, device=dev)).double().flip(-1)  # desc
        ev = ev.clamp_min(0)
        nval = valid.sum(-1).double()
        evn = ev / nval[:, None]
        top = evn[:, :N_EIG]
        pr = evn / evn.sum(-1, keepdim=True).clamp_min(1e-12)
        erank = torch.exp(-(pr * torch.log(pr.clamp_min(1e-12))).sum(-1)) / nval
        r12 = sc.rlead[ar, 0, sc.lead2].double()
    else:
        Rmm = None
        top = torch.zeros(B, N_EIG, dtype=torch.float64, device=dev)
        erank = torch.zeros(B, dtype=torch.float64, device=dev)
        r12 = torch.zeros(B, dtype=torch.float64, device=dev)
    pw = mask.sum(-1).double()
    med = lambda x: torch.where(mask, x, torch.nan).nanmedian(-1).values
    mz1 = z1.abs().masked_fill(~mask, 0).amax(-1)
    mz2 = z2.abs().masked_fill(~mask, 0).amax(-1)
    glob_raw = torch.cat([torch.stack([
        b["rho_hat"].double(), b["rho_se"].double().clamp(max=1.0), med(ln1) / 5, med(ln2) / 5, torch.log10(pw) / 4,
        torch.log10(n_ref.double().clamp_min(1.0)) / 5, b["type1"].double(), b["type2"].double(), mz1, mz2,
        torch.log1p(((z1.abs() > 5.45) & mask).sum(-1).double()), torch.log1p(((z2.abs() > 5.45) & mask).sum(-1).double()),
        erank, r12, torch.sign(z1[ar, sc.lead1]), torch.sign(z2[ar, sc.lead2]),
        (sc.lead1 == sc.lead2).double(), torch.log1p(pw - valid.sum(-1).double()) / 8], -1), top], -1)
    return dict(tok_raw=tok_raw, src=sc.source, valid=valid, Rmm=Rmm, all_raw=all_raw, mask=mask, glob_raw=glob_raw,
                sel=sel, lead1=sc.lead1, lead2=sc.lead2)


GLOB_RAW_RHO, GLOB_RAW_MZ, GLOB_RAW_R12, GLOB_RAW_SIGN = 0, 8, 13, 14


@torch.no_grad()
def build(rec: dict, flip_traits: torch.Tensor | None = None) -> dict:
    """Stage 2 (no LD access): trait-flip augmentation, allele canonicalisation, scaled model inputs.

    flip_traits [B, 2] in {+1, -1}: z_t -> s_t z_t (conditional z likewise), rho -> s_1 s_2 rho."""
    tr = rec["tok_raw"].double()
    ar_ = rec["all_raw"].double()
    gr = rec["glob_raw"].double()
    valid, mask = rec["valid"], rec["mask"]
    B, M, _ = tr.shape
    dev = tr.device
    f1 = torch.ones(B, dtype=torch.float64, device=dev)
    f2 = torch.ones(B, dtype=torch.float64, device=dev)
    if flip_traits is not None:
        f1, f2 = flip_traits[:, 0].double(), flip_traits[:, 1].double()
    s1, s2 = tr[..., 0] * f1[:, None], tr[..., 1] * f2[:, None]
    zc1, zc2 = tr[..., 2] * f1[:, None], tr[..., 3] * f2[:, None]
    sgn = torch.where(s1.abs() >= s2.abs(), torch.sign(s1), torch.sign(s2))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    c1, c2 = sgn * s1, sgn * s2
    lead_s1 = gr[:, GLOB_RAW_SIGN] * f1
    lead_s2 = gr[:, GLOB_RAW_SIGN + 1] * f2
    rl1 = tr[..., 4] * sgn * lead_s1[:, None]
    rl2 = tr[..., 5] * sgn * lead_s2[:, None]
    onehot = torch.zeros(B, M, 5, dtype=torch.float64, device=dev)
    onehot.scatter_(2, rec["src"].long().clamp_min(0)[..., None], 1.0)
    tok = torch.cat([torch.stack([
        _asinh2(c1), _asinh2(c2), c1 / 20, c2 / 20, _asinh2(sgn * zc1), _asinh2(sgn * zc2), rl1, rl2, rl1 ** 2,
        rl2 ** 2, tr[..., 6] / 5, tr[..., 7] / 5, tr[..., 8] * 2, tr[..., 9] * 2, tr[..., 10] * 2, tr[..., 11],
        tr[..., 12]], -1), onehot], -1) * valid[..., None]
    assert tok.shape[-1] == N_TOK_FEAT

    def quad(a1, a2, b1, b2):
        return torch.stack([a1 * a1, a2 * a2, a1 * a2, b1 * b1, b2 * b2, b1 * b2], -1) / 2.0
    q_tok = quad(c1, c2, sgn * zc1, sgn * zc2) * valid[..., None]
    z1a, z2a = ar_[..., 0] * f1[:, None], ar_[..., 1] * f2[:, None]
    q_all = quad(z1a, z2a, ar_[..., 2] * f1[:, None], ar_[..., 3] * f2[:, None]) * mask[..., None]
    all_feat = torch.stack([ar_[..., 4] / 5, ar_[..., 5] / 5, ar_[..., 6] / 5, ar_[..., 7] / 5, ar_[..., 8] * 2,
                            ar_[..., 9] * 2, ar_[..., 10] * 2, ar_[..., 11], ar_[..., 12], torch.zeros_like(z1a)],
                           -1) * mask[..., None]
    Rmm = rec["Rmm"]
    if Rmm is not None:
        Rmm = Rmm.double() * sgn[:, :, None] * sgn[:, None, :]
    rho = gr[:, GLOB_RAW_RHO] * f1 * f2
    mz1, mz2 = gr[:, GLOB_RAW_MZ], gr[:, GLOB_RAW_MZ + 1]
    r12 = gr[:, GLOB_RAW_R12] * lead_s1 * lead_s2
    glob = torch.cat([torch.stack([
        rho, gr[:, 1], gr[:, 2], gr[:, 3], gr[:, 4], gr[:, 5], gr[:, 6], gr[:, 7], _asinh2(mz1), _asinh2(mz2),
        mz1 / 50, mz2 / 50, gr[:, 10], gr[:, 11], gr[:, 12], r12, r12 ** 2, gr[:, 16], gr[:, 17], rho * rho], -1),
        gr[:, 18:]], -1)
    assert glob.shape[-1] == N_GLOB_FEAT, glob.shape
    return dict(tok=tok.float(), valid=valid, Rmm=None if Rmm is None else Rmm.float(), q_tok=q_tok.float(),
                q_all=q_all.float(), all_feat=all_feat.float(), mask=mask, glob=glob.float(), sel=rec["sel"],
                sgn=sgn, lead1=rec["lead1"], lead2=rec["lead2"])


def _screen_nold(z1, z2, mask, M):
    """No-LD ablation screen: marginal lists only (quotas M/2, M/2), round-robin, no LD rows."""
    B, P = z1.shape
    dev = z1.device
    neg = torch.tensor(-math.inf, dtype=z1.dtype, device=dev)
    a1 = torch.where(mask, z1.abs(), neg)
    a2 = torch.where(mask, z2.abs(), neg)
    lead1, lead2 = a1.argmax(-1), a2.argmax(-1)
    sel, src = _merge_lists(torch.stack([a1, a2], 0), mask, [M // 2, M - M // 2], M)
    pw = mask.sum(-1)
    small = pw <= M
    valid = torch.ones(B, M, dtype=torch.bool, device=dev)
    if small.any():
        idx = torch.arange(M, device=dev)[None, :].expand(int(small.sum()), M)
        v = idx < pw[small][:, None]
        sel[small] = torch.where(v, idx, torch.zeros_like(idx))
        src[small] = torch.where(v, torch.full_like(idx, 4), torch.full_like(idx, -1))
        valid[small] = v
    z0 = torch.zeros_like(z1)
    return ScreenOut(sel, valid, src, lead1, lead2, torch.zeros(B, 2, P, dtype=z1.dtype, device=dev), z0, z0)

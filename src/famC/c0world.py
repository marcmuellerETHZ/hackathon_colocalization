"""C0 kill-test world (DESIGN v4 §5 C, "C0"): coloc.abf's own world on real 1000G LD windows.

Per locus (all on the GPU, batched):
  * a window of the LD bank (real pool-A genotypes, n = 254; train split for training, validation split for
    evaluation); R = X'X/n;
  * sample sizes N1, N2 ~ log-uniform [N_LO, N_HI] independently; se_tj = 1/sqrt(2 N_t f_j (1 - f_j)) with f the
    pool allele frequency (sdY = 1, DESIGN §2.2);
  * hypothesis H in {H0..H4} (coloc's): H1 trait 1 only, H2 trait 2 only, H3 two distinct variants, H4 one shared;
    at most one causal variant per trait, positions uniform over the window's SNPs (coloc's uniform prior);
  * effects beta_t ~ N(0, W^2) (coloc's prior effect SD W), noncentrality lambda_t = beta_t / se_t,causal;
  * z_t = lambda_t R e_j + X' u_t / sqrt(n), u_t ~ N(0, I_n) independent between traits, so z_t ~ N(R e_j lambda_t, R);
  * target = coloc.abf posterior PP(H0..H4) at (W, p1, p2, p12) computed from (z, se) over the full window, and
    SNP.PP.H4.
`hyp_probs = "coloc"` draws H from coloc's own prior at the window's p (adjusted as coloc does); a vector of 5
probabilities draws from that instead (training uses uniform, README).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from famC.ldbank import LDBank, logu

W_DEFAULT = 0.15
P1, P2, P12 = 1e-4, 1e-4, 1e-5
N_LO, N_HI = 300.0, 300_000.0


def _adjust(p: float, n: torch.Tensor) -> torch.Tensor:
    """coloc adjust_prior: if nsnps * p >= 1 then p = 1 / (nsnps + 1)."""
    n = n.double()
    return torch.where(n * p >= 1, 1.0 / (n + 1), torch.full_like(n, p))


def coloc_prior_probs(p: torch.Tensor, p1=P1, p2=P2, p12=P12) -> torch.Tensor:
    """coloc's implied prior over H0..H4 for windows with p SNPs. [B, 5]

    coloc.abf sets lH0 = 0 and lH1 = log p1 + log sum_j BF1_j etc., i.e. prior odds H0 : H1 : H2 : H3 : H4 =
    1 : p a1 : p a2 : p (p - 1) a1 a2 : p a12 (a = adjusted per-SNP priors); normalised here."""
    a1, a2, a12 = _adjust(p1, p), _adjust(p2, p), _adjust(p12, p)
    p = p.double()
    out = torch.stack([torch.ones_like(p), p * a1, p * a2, p * (p - 1) * a1 * a2, p * a12], -1)
    return out / out.sum(-1, keepdim=True)


def labf_t(z: torch.Tensor, se: torch.Tensor, W: float) -> torch.Tensor:
    V = se.double() ** 2
    r = W * W / (W * W + V)
    return 0.5 * (torch.log1p(-r) + r * z.double() ** 2)


def coloc_abf_torch(z1, se1, z2, se2, mask, W=W_DEFAULT, p1=P1, p2=P2, p12=P12):
    """Batched coloc.abf (fp64), identical formulas to baselines.abf.coloc_abf. Returns PP [B,5], SNP.PP.H4 [B,P]."""
    l1 = labf_t(z1, se1, W).masked_fill(~mask, -math.inf)
    l2 = labf_t(z2, se2, W).masked_fill(~mask, -math.inf)
    n = mask.sum(-1)
    a1, a2, a12 = _adjust(p1, n), _adjust(p2, n), _adjust(p12, n)
    ls1, ls2 = torch.logsumexp(l1, -1), torch.logsumexp(l2, -1)
    s = l1 + l2
    ls12 = torch.logsumexp(s, -1)
    x, y = ls1 + ls2, ls12
    m = torch.maximum(x, y)
    ld = m + torch.log(torch.exp(x - m) - torch.exp(y - m))
    lH = torch.stack([torch.zeros_like(ls1), torch.log(a1) + ls1, torch.log(a2) + ls2,
                      torch.log(a1) + torch.log(a2) + ld, torch.log(a12) + ls12], -1)
    pp = torch.softmax(lH, -1)
    snp4 = torch.softmax(s, -1)
    return pp, snp4


@torch.no_grad()
def sample_c0(bank: LDBank, g: torch.Generator, gcpu: torch.Generator, n_win: int, per_win: int,
              hyp_probs="uniform", W: float = W_DEFAULT, n_range=(N_LO, N_HI), widx=None) -> dict:
    """One batch of C0 loci. g: CUDA generator (all draws), gcpu: CPU generator (window choice).

    Returns dict with z1, z2, se1, se2, n1, n2, maf [B,P] (padding se=1, z=0), mask [B,P], X [Wb,n,P],
    win_of [B], pp [B,5] (target), snp_pp4 [B,P], hyp [B], causal1/2 [B] (-1 if none), lam1/2 [B].
    """
    dev = bank.device
    if widx is None:
        widx = bank.sample_windows(gcpu, n_win)
    Wb = len(widx)
    X, wmask = bank.standardized(widx)                       # [Wb, n, P]
    af = bank.af_padded(widx)                                # [Wb, P]
    n = X.shape[1]
    P = X.shape[2]
    L = per_win
    B = Wb * L
    win_of = torch.arange(Wb, device=dev).repeat_interleave(L)
    mask = wmask[win_of]                                     # [B, P]
    pw = mask.sum(-1)                                        # [B]
    f = af[win_of]
    N1 = logu(g, *n_range, (B,), dev)
    N2 = logu(g, *n_range, (B,), dev)
    v = 2.0 * f * (1 - f)
    se1 = torch.where(mask, 1.0 / torch.sqrt(N1[:, None] * v), torch.ones_like(v))
    se2 = torch.where(mask, 1.0 / torch.sqrt(N2[:, None] * v), torch.ones_like(v))
    # hypotheses
    if isinstance(hyp_probs, str) and hyp_probs == "coloc":
        probs = coloc_prior_probs(pw)
    elif isinstance(hyp_probs, str) and hyp_probs == "uniform":
        probs = torch.full((B, 5), 0.2, dtype=torch.float64, device=dev)
    else:
        probs = torch.as_tensor(hyp_probs, dtype=torch.float64, device=dev).expand(B, 5)
    hyp = torch.multinomial(probs.float(), 1, generator=g).squeeze(-1)
    u = torch.rand(B, generator=g, device=dev, dtype=torch.float64)
    j1 = torch.floor(u * pw).long()
    u2 = torch.rand(B, generator=g, device=dev, dtype=torch.float64)
    k = torch.floor(u2 * (pw - 1)).long()
    j2d = k + (k >= j1).long()                               # uniform over the others
    has1 = (hyp == 1) | (hyp == 3) | (hyp == 4)
    has2 = (hyp == 2) | (hyp == 3) | (hyp == 4)
    c1 = torch.where(has1, j1, torch.full_like(j1, -1))
    c2 = torch.where(hyp == 4, j1, torch.where(hyp == 3, j2d, torch.where(hyp == 2, j1, torch.full_like(j1, -1))))
    b1 = torch.randn(B, generator=g, device=dev, dtype=torch.float64) * W
    b2 = torch.randn(B, generator=g, device=dev, dtype=torch.float64) * W
    ar = torch.arange(B, device=dev)
    lam1 = torch.where(has1, b1 / se1[ar, c1.clamp_min(0)], torch.zeros_like(b1))
    lam2 = torch.where(has2, b2 / se2[ar, c2.clamp_min(0)], torch.zeros_like(b2))
    # noise and signal, grouped by window: [Wb, L, n] @ [Wb, n, P]
    U1 = torch.randn(Wb, L, n, generator=g, device=dev)
    U2 = torch.randn(Wb, L, n, generator=g, device=dev)
    e1 = torch.bmm(U1, X) / math.sqrt(n)
    e2 = torch.bmm(U2, X) / math.sqrt(n)
    Xc1 = torch.gather(X, 2, c1.clamp_min(0).view(Wb, 1, L).expand(Wb, n, L))  # [Wb, n, L]
    Xc2 = torch.gather(X, 2, c2.clamp_min(0).view(Wb, 1, L).expand(Wb, n, L))
    Rc1 = torch.bmm(Xc1.transpose(1, 2), X) / n                 # [Wb, L, P] = R[:, c]
    Rc2 = torch.bmm(Xc2.transpose(1, 2), X) / n
    z1 = (e1.reshape(B, P).double() + lam1[:, None] * Rc1.reshape(B, P).double())
    z2 = (e2.reshape(B, P).double() + lam2[:, None] * Rc2.reshape(B, P).double())
    z1 = torch.where(mask, z1, torch.zeros_like(z1))
    z2 = torch.where(mask, z2, torch.zeros_like(z2))
    pp, snp4 = coloc_abf_torch(z1, se1, z2, se2, mask, W=W)
    maf = torch.minimum(f, 1 - f)
    return dict(z1=z1, z2=z2, se1=se1, se2=se2, n1=N1[:, None].expand(B, P), n2=N2[:, None].expand(B, P),
                maf1=maf, maf2=maf, maf_ref=maf, mask=mask, X=X, win_of=win_of, n_ref=n,
                rho_hat=torch.zeros(B, dtype=torch.float64, device=dev),
                rho_se=torch.zeros(B, dtype=torch.float64, device=dev),
                type1=torch.zeros(B, device=dev), type2=torch.ones(B, device=dev),
                pp=pp, snp_pp4=snp4, hyp=hyp, causal1=c1, causal2=c2, lam1=lam1, lam2=lam2,
                widx=np.asarray(widx))

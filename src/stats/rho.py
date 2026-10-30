"""rho-hat: truncated bivariate-normal MLE of the null z-score correlation (DESIGN v4 §2.4, decision D6(d)).

Model: for SNPs without signal, (z1, z2) ~ N(0, [[1, rho], [rho, 1]]). Only SNPs with |z1| < c and |z2| < c
(c = 2) are used, so each SNP contributes the bivariate normal density truncated to the square:

    log L(rho) = sum_i log phi2(z1i, z2i; rho) - n log P_rho(|Z1| < c, |Z2| < c)

(a composite likelihood when SNPs are in LD: the estimating equation stays unbiased, the SE must account for
the correlation). The likelihood depends on the data only through (n, sum z1² + z2², sum z1 z2).

D6(d) estimator (one estimator for simulation and real data):
  1. signal mask: every SNP in LD (reference r² > 0.1) with ANY SNP whose max(|z1|, |z2|) > 4 is excluded
     (the signal SNPs themselves included);
  2. truncated MLE on the remaining SNPs with |z1| < 2 and |z2| < 2 (no LD pruning: the jackknife carries
     the LD-induced dependence);
  3. SE by delete-one-block jackknife over contiguous blocks of the used SNPs (position order), equal SNP
     counts per block (`n_blocks`, README); across windows (`rho_hat_pooled`) the windows are the blocks.

The MLE is solved on a fixed grid of rho (step 5e-4 on (-0.995, 0.995)) with parabolic refinement, which
is vectorised over many (n, sqq, s12) triples (jackknife replicates, batches); `truncated_mle_brent` is the
exact scalar reference used in the tests (agreement < 1e-6).

Modes (DESIGN §2.4):
  per-window: `rho_hat_window(z1, z2, R or X)`: mask with the LD given to methods, MLE, block jackknife;
  genome-wide: `rho_hat_genomewide(rho, rng, ...)`: 20,000 independent SNPs (null + 1% polygenic signal,
               independent effects in the two traits) drawn with the study's true rho, same estimator;
  pooled: `rho_hat_pooled([(z1, z2, R), ...])`: one cohort pair over many windows (real data, D6(d)).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)
RHO_BOUND = 0.995
C_TRUNC = 2.0
Z_SIGNAL = 4.0
MASK_R2 = 0.1
PRUNE_R2 = 0.1
N_BLOCKS = 10
MIN_N = 10
GRID_STEP = 5e-4


def square_prob(rho, c: float = C_TRUNC):
    """P(|Z1| < c, |Z2| < c) for a standard bivariate normal with correlation rho (scalar or array)."""
    rho = np.asarray(rho, dtype=np.float64)
    x = _GL_X * c
    w = _GL_W * c
    s = np.sqrt(1.0 - rho[..., None] ** 2)
    phi = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
    inner = ndtr((c - rho[..., None] * x) / s) - ndtr((-c - rho[..., None] * x) / s)
    out = np.sum(w * phi * inner, axis=-1)
    return float(out) if out.ndim == 0 else out


def _negll(rho, n, sqq, s12, c: float = C_TRUNC):
    """Negative log-likelihood (per the sufficient statistics); broadcasts over rho and the stats."""
    rho = np.asarray(rho, dtype=np.float64)
    one = 1.0 - rho * rho
    ll = -n * np.log(2 * np.pi) - 0.5 * n * np.log(one) - (sqq - 2 * rho * s12) / (2 * one)
    return -(ll - n * np.log(square_prob(rho, c)))


@lru_cache(maxsize=4)
def _grid(c: float):
    r = np.arange(-RHO_BOUND, RHO_BOUND + 1e-12, GRID_STEP)
    one = 1.0 - r * r
    # negll / n = A(r) + B(r) * (sqq/n) - C(r) * (s12/n)
    A = np.log(2 * np.pi) + 0.5 * np.log(one) + np.log(square_prob(r, c))
    B = 1.0 / (2 * one)
    C = r / one
    return r, A, B, C


def mle_from_stats(n, sqq, s12, c: float = C_TRUNC) -> np.ndarray:
    """Vectorised MLE of rho from sufficient statistics (arrays of equal shape). NaN where n == 0."""
    n = np.asarray(n, dtype=np.float64)
    sqq = np.asarray(sqq, dtype=np.float64)
    s12 = np.asarray(s12, dtype=np.float64)
    shape = np.broadcast(n, sqq, s12).shape
    n, sqq, s12 = (np.broadcast_to(a, shape).ravel() for a in (n, sqq, s12))
    r, A, B, C = _grid(float(c))
    out = np.full(n.shape, np.nan)
    ok = n > 0
    if ok.any():
        a = sqq[ok] / n[ok]
        b = s12[ok] / n[ok]
        f = A[None, :] + B[None, :] * a[:, None] - C[None, :] * b[:, None]
        k = np.argmin(f, axis=1)
        k = np.clip(k, 1, len(r) - 2)
        i = np.arange(len(k))
        f0, f1, f2 = f[i, k - 1], f[i, k], f[i, k + 1]
        den = f0 - 2 * f1 + f2
        off = np.where(den > 0, 0.5 * (f0 - f2) / np.where(den > 0, den, 1.0), 0.0)
        out[ok] = np.clip(r[k] + np.clip(off, -1, 1) * GRID_STEP, -RHO_BOUND, RHO_BOUND)
    return out.reshape(shape)


# ------------------------------------------------------------------------------------------- 3-parameter model
def rect_prob(a, b, rho):
    """P(|Z1| < a, |Z2| < b) for a standard bivariate normal with correlation rho (scalars)."""
    x = _GL_X * a
    w = _GL_W * a
    s = np.sqrt(1.0 - rho * rho)
    phi = np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
    inner = ndtr((b - rho * x) / s) - ndtr((-b - rho * x) / s)
    return float(np.sum(w * phi * inner))


def _negll3(theta, n, S11, S22, S12, c):
    l1, l2, t = theta
    s1, s2, r = np.exp(l1), np.exp(l2), np.tanh(t)
    one = 1.0 - r * r
    q = (S11 / s1 ** 2 - 2 * r * S12 / (s1 * s2) + S22 / s2 ** 2) / one
    P = max(rect_prob(c / s1, c / s2, r), 1e-300)
    return n * (np.log(2 * np.pi) + l1 + l2 + 0.5 * np.log(one) + np.log(P)) + 0.5 * q


def mle3_from_stats(n, S11, S22, S12, c: float = C_TRUNC, x0=None) -> tuple[float, float, float]:
    """Truncated bivariate-normal MLE with free marginal SDs: (rho, sigma1, sigma2). Robust to null-variance
    inflation (e.g. weak LD-tagged signal), which biases the unit-variance model towards 0 (README D14)."""
    from scipy.optimize import minimize
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    r0 = S12 / np.sqrt(S11 * S22) if S11 > 0 and S22 > 0 else 0.0
    x0 = np.array([0.0, 0.0, np.arctanh(np.clip(r0, -0.9, 0.9))]) if x0 is None else np.asarray(x0, float)
    res = minimize(_negll3, x0, args=(n, S11, S22, S12, c), method="L-BFGS-B",
                   bounds=[(-1.5, 1.5), (-1.5, 1.5), (-np.arctanh(RHO_BOUND), np.arctanh(RHO_BOUND))],
                   options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 500})
    l1, l2, t = res.x
    return float(np.tanh(t)), float(np.exp(l1)), float(np.exp(l2))


def info_se3(n, S11, S22, S12, c: float = C_TRUNC, est=None) -> float:
    """Observed-information SE of rho in the 3-parameter model (numerical Hessian in (log s1, log s2, atanh r),
    delta method to rho), assuming independent SNPs."""
    r, s1, s2 = est if est is not None else mle3_from_stats(n, S11, S22, S12, c)
    th = np.array([np.log(s1), np.log(s2), np.arctanh(np.clip(r, -RHO_BOUND, RHO_BOUND))])
    h = 1e-4
    H = np.zeros((3, 3))
    f0 = _negll3(th, n, S11, S22, S12, c)
    for i in range(3):
        for j in range(i, 3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            H[i, j] = H[j, i] = (_negll3(th + ei + ej, n, S11, S22, S12, c) - _negll3(th + ei - ej, n, S11, S22, S12, c)
                                 - _negll3(th - ei + ej, n, S11, S22, S12, c) + _negll3(th - ei - ej, n, S11, S22, S12, c)) / (4 * h * h)
    try:
        V = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("inf")
    v = V[2, 2]
    return float(np.sqrt(v) * (1 - r * r)) if v > 0 else float("inf")


def block_stats4(z1, z2, use: np.ndarray, n_blocks: int, blocks=None):
    """Per-block (n, S11, S22, S12) of the used SNPs (contiguous equal-count blocks or given labels)."""
    idx = np.where(use)[0]
    a, b = np.asarray(z1)[idx], np.asarray(z2)[idx]
    if blocks is None:
        nb = max(1, min(n_blocks, len(idx)))
        lab = (np.arange(len(idx)) * nb) // max(len(idx), 1)
    else:
        _, lab = np.unique(np.asarray(blocks)[idx], return_inverse=True)
        nb = int(lab.max()) + 1 if len(lab) else 0
    return (np.bincount(lab, minlength=nb).astype(np.float64), np.bincount(lab, weights=a * a, minlength=nb),
            np.bincount(lab, weights=b * b, minlength=nb), np.bincount(lab, weights=a * b, minlength=nb))


def jackknife3(n, S11, S22, S12, c: float = C_TRUNC) -> tuple[float, float, float, float]:
    """3-parameter MLE on all blocks + delete-one-block jackknife SE of rho: (rho, se, s1, s2)."""
    full = mle3_from_stats(n.sum(), S11.sum(), S22.sum(), S12.sum(), c)
    B = len(n)
    if B < 2:
        return full[0], float("nan"), full[1], full[2]
    x0 = np.array([np.log(full[1]), np.log(full[2]), np.arctanh(np.clip(full[0], -0.99, 0.99))])
    loo = np.array([mle3_from_stats(n.sum() - n[b], S11.sum() - S11[b], S22.sum() - S22[b], S12.sum() - S12[b],
                                    c, x0=x0)[0] for b in range(B)])
    se = float(np.sqrt((B - 1) / B * np.sum((loo - loo.mean()) ** 2)))
    return full[0], se, full[1], full[2]


def truncated_mle_brent(z1, z2, c: float = C_TRUNC, min_n: int = MIN_N) -> tuple[float, float, int]:
    """Exact scalar reference: (rho_hat, observed-information SE, n_used) by bounded Brent search."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    m = (np.abs(z1) < c) & (np.abs(z2) < c) & np.isfinite(z1) & np.isfinite(z2)
    n = int(m.sum())
    if n < min_n:
        return float("nan"), float("nan"), n
    a, b = z1[m], z2[m]
    sqq = float(np.sum(a * a + b * b))
    s12 = float(np.sum(a * b))
    res = minimize_scalar(lambda x: float(_negll(x, n, sqq, s12, c)), bounds=(-RHO_BOUND, RHO_BOUND),
                          method="bounded", options={"xatol": 1e-10})
    return float(res.x), _info_se(float(res.x), n, sqq, s12, c), n


def _info_se(r: float, n, sqq, s12, c) -> float:
    h = 1e-4
    lo, hi = max(-RHO_BOUND, r - h), min(RHO_BOUND, r + h)
    mid, hh = 0.5 * (lo + hi), 0.5 * (hi - lo)
    f2 = (_negll(mid + hh, n, sqq, s12, c) - 2 * _negll(mid, n, sqq, s12, c) + _negll(mid - hh, n, sqq, s12, c)) / hh ** 2
    return float(1.0 / np.sqrt(f2)) if f2 > 0 else float("inf")


def truncated_mle(z1, z2, c: float = C_TRUNC, min_n: int = MIN_N) -> tuple[float, float, int]:
    """(rho_hat, observed-information SE assuming independent SNPs, n_used) from |z| < c SNPs."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    m = (np.abs(z1) < c) & (np.abs(z2) < c) & np.isfinite(z1) & np.isfinite(z2)
    n = int(m.sum())
    if n < min_n:
        return float("nan"), float("nan"), n
    sqq = float(np.sum(z1[m] ** 2 + z2[m] ** 2))
    s12 = float(np.sum(z1[m] * z2[m]))
    r = float(mle_from_stats(n, sqq, s12, c))
    return r, _info_se(r, n, sqq, s12, c), n


# ------------------------------------------------------------------------------------------- masking
def _ld_rows_fn(ld):
    """ld: None (independent SNPs), dense R [p, p], ('factor', X [n, p]) or an object with ld_rows(idx)."""
    if ld is None:
        return None
    if hasattr(ld, "ld_rows"):
        return ld.ld_rows
    if isinstance(ld, tuple) and ld[0] == "factor":
        X = np.asarray(ld[1], dtype=np.float64)
        n = X.shape[0]
        return lambda idx: X[:, idx].T @ X / n
    R = ld
    return lambda idx: np.asarray(R[np.asarray(idx)], dtype=np.float64)


def signal_mask(z1, z2, ld=None, r2_mask: float = MASK_R2, z_signal: float = Z_SIGNAL) -> np.ndarray:
    """Boolean [p]: True = excluded (a signal SNP, max(|z1|,|z2|) > z_signal, or in LD r² > r2_mask with one)."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    sig = np.where(np.maximum(np.abs(z1), np.abs(z2)) > z_signal)[0]
    out = np.zeros(len(z1), bool)
    out[sig] = True
    rows = _ld_rows_fn(ld)
    if rows is not None and len(sig):
        for a in range(0, len(sig), 256):
            Rr = rows(sig[a:a + 256])
            out |= np.any(Rr * Rr > r2_mask, axis=0)
    return out


def residualize(z1, z2, ld, sig: np.ndarray, max_cond: int = 30, r2_cond: float = 0.8, ridge: float = 1e-3):
    """Remove the LD-propagated part of strong signals from both traits (README D14): a conditioning set C is
    chosen greedily from the signal SNPs `sig` by decreasing max(|z1|,|z2|) (r² < r2_cond among C, <= max_cond),
    then z_res,k = (z_k - R_kC R_CC^-1 z_C) / sqrt(1 - R_kC R_CC^-1 R_Ck) for both traits with the same C.
    Under the null this keeps N(0,1) margins and the cross-trait correlation rho exactly (with the true R), and
    the residuals are independent of z_C, so selecting C from the data does not bias them."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    rows = _ld_rows_fn(ld)
    if rows is None or len(sig) == 0:
        return z1, z2, 0
    order = np.asarray(sig)[np.argsort(-np.maximum(np.abs(z1[sig]), np.abs(z2[sig])), kind="stable")]
    C: list[int] = []
    Rc = []
    for j in order:
        rj = np.asarray(rows([int(j)])[0], dtype=np.float64)
        if all(rj[c] ** 2 < r2_cond for c in C):
            C.append(int(j))
            Rc.append(rj)
            if len(C) >= max_cond:
                break
    Rc = np.asarray(Rc)                                   # [k, p]
    Rcc = Rc[:, C] + ridge * np.eye(len(C))
    W = np.linalg.solve(Rcc, Rc)                          # R_CC^-1 R_C.  [k, p]
    v = 1.0 - np.einsum("kp,kp->p", Rc, W)
    v = np.clip(v, 1e-3, None)
    a1 = (z1 - W.T @ z1[C]) / np.sqrt(v)
    a2 = (z2 - W.T @ z2[C]) / np.sqrt(v)
    return a1, a2, len(C)


# ------------------------------------------------------------------------------------------- jackknife
def block_stats(z1, z2, use: np.ndarray, n_blocks: int, blocks=None):
    """Per-block (n, sqq, s12) of the used SNPs. Blocks: contiguous equal-count chunks of the used SNPs in
    array order, or the given integer labels `blocks` (e.g. window index)."""
    idx = np.where(use)[0]
    a, b = np.asarray(z1)[idx], np.asarray(z2)[idx]
    if blocks is None:
        nb = max(1, min(n_blocks, len(idx)))
        lab = (np.arange(len(idx)) * nb) // max(len(idx), 1)
    else:
        _, lab = np.unique(np.asarray(blocks)[idx], return_inverse=True)
        nb = int(lab.max()) + 1 if len(lab) else 0
    n = np.bincount(lab, minlength=nb).astype(np.float64)
    sqq = np.bincount(lab, weights=a * a + b * b, minlength=nb)
    s12 = np.bincount(lab, weights=a * b, minlength=nb)
    return n, sqq, s12


def jackknife_from_blocks(n, sqq, s12, c: float = C_TRUNC) -> tuple[float, float]:
    """(full-data MLE, delete-one-block jackknife SE). SE NaN with < 2 blocks."""
    N, Q, S = n.sum(), sqq.sum(), s12.sum()
    full = float(mle_from_stats(N, Q, S, c))
    B = len(n)
    if B < 2:
        return full, float("nan")
    loo = mle_from_stats(N - n, Q - sqq, S - s12, c)
    se = float(np.sqrt((B - 1) / B * np.sum((loo - loo.mean()) ** 2)))
    return full, se


def estimate(z1, z2, ld=None, c: float = C_TRUNC, r2_mask: float = MASK_R2, z_signal: float = Z_SIGNAL,
             n_blocks: int = N_BLOCKS, min_n: int = MIN_N, mask: np.ndarray | None = None, blocks=None) -> dict:
    """The D6(d) estimator on one set of SNPs (array order = genomic order)."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    if mask is None:
        mask = signal_mask(z1, z2, ld, r2_mask, z_signal)
    use = (~mask) & (np.abs(z1) < c) & (np.abs(z2) < c) & np.isfinite(z1) & np.isfinite(z2)
    n_used = int(use.sum())
    out = {"n_used": n_used, "n_masked": int(mask.sum()), "n_total": int(len(z1))}
    if n_used < min_n:
        out.update(rho_hat=float("nan"), rho_se=float("nan"), n_blocks=0)
        return out
    n, sqq, s12 = block_stats(z1, z2, use, n_blocks, blocks)
    r, se = jackknife_from_blocks(n, sqq, s12, c)
    out.update(rho_hat=r, rho_se=se, n_blocks=int(len(n)),
               se_info=_info_se(r, n.sum(), sqq.sum(), s12.sum(), c))
    return out


def rho_hat_window(z1, z2, R=None, r2_max: float = PRUNE_R2, c: float = C_TRUNC, order=None, *,
                   z_signal: float = Z_SIGNAL, mask_r2: float = MASK_R2, n_blocks: int = N_BLOCKS,
                   residualize_signal: bool = False) -> dict:
    """Per-window estimator (README D14; D6(d) + DESIGN §2.4 pruning):
      1. signal SNPs: max(|z1|,|z2|) > z_signal; mask = signal SNPs + SNPs with r² > mask_r2 to one of them;
      2. optional (residualize_signal, only with accurate LD: in-sample or n_ref >= 1000): project the signal
         SNPs' LD-propagated part out of both traits (`residualize`);
      3. greedy LD pruning (r² < r2_max) of ALL window SNPs in `order` (z-independent), then keep the pruned,
         unmasked SNPs with |z| < c in both traits;
      4. unit-variance truncated MLE; SE = delete-one-block jackknife over n_blocks contiguous blocks.
    R: dense LD [p, p] (the LD given to methods); `order`: pruning visit order (default: position order)."""
    z1 = np.asarray(z1, dtype=np.float64)
    z2 = np.asarray(z2, dtype=np.float64)
    R = np.asarray(R)
    sig = np.where(np.maximum(np.abs(z1), np.abs(z2)) > z_signal)[0]
    mask = signal_mask(z1, z2, R, mask_r2, z_signal)
    n_cond = 0
    if residualize_signal and len(sig):
        z1, z2, n_cond = residualize(z1, z2, R, sig)
    keep = prune(R, r2_max, order)
    pm = np.zeros(len(z1), bool)
    pm[keep] = True
    out = estimate(z1, z2, None, c, mask_r2, z_signal, n_blocks, mask=mask | ~pm)
    out["n_masked"] = int(mask.sum())
    out.update(mode="window", n_pruned=int(len(keep)), n_cond=int(n_cond), residualized=bool(residualize_signal))
    return out


def rho_hat_pooled(windows, c: float = C_TRUNC, r2_mask: float = MASK_R2, z_signal: float = Z_SIGNAL,
                   min_windows_as_blocks: int = 10, n_blocks: int = N_BLOCKS) -> dict:
    """One cohort pair over many windows [(z1, z2, ld), ...] (D6(d) real-data pooled mode). Masks are
    computed per window; jackknife blocks are the windows (or contiguous blocks if < 10 windows)."""
    Z1, Z2, M, W = [], [], [], []
    for k, (a, b, ld) in enumerate(windows):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        Z1.append(a)
        Z2.append(b)
        M.append(signal_mask(a, b, ld, r2_mask, z_signal))
        W.append(np.full(len(a), k))
    z1, z2, mask, lab = np.concatenate(Z1), np.concatenate(Z2), np.concatenate(M), np.concatenate(W)
    use_w = len(windows) >= min_windows_as_blocks
    out = estimate(z1, z2, None, c, r2_mask, z_signal, n_blocks, mask=mask, blocks=lab if use_w else None)
    out["mode"] = "pooled"
    out["n_windows"] = len(windows)
    return out


# ------------------------------------------------------------------------------------------- genome-wide
def simulate_genomewide_z(rho: float, g: np.random.Generator, n_snp: int = 20_000, poly: float = 0.01,
                          poly_sd: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Independent SNP z-scores with null correlation rho; a fraction `poly` carries independent effects."""
    e1 = g.standard_normal(n_snp)
    e2 = rho * e1 + np.sqrt(1 - rho * rho) * g.standard_normal(n_snp)
    sig = g.random(n_snp) < poly
    m1 = np.where(sig, g.normal(0, poly_sd, n_snp), 0.0)
    m2 = np.where(sig, g.normal(0, poly_sd, n_snp), 0.0)
    return m1 + e1, m2 + e2


def rho_hat_genomewide(rho: float, g: np.random.Generator, n_snp: int = 20_000, poly: float = 0.01,
                       poly_sd: float = 2.0, c: float = C_TRUNC, n_blocks: int = N_BLOCKS) -> dict:
    z1, z2 = simulate_genomewide_z(rho, g, n_snp, poly, poly_sd)
    out = estimate(z1, z2, None, c, MASK_R2, Z_SIGNAL, n_blocks)
    out["mode"] = "genomewide"
    return out


# ------------------------------------------------------------------------------------------- legacy
def prune(R: np.ndarray, r2_max: float = 0.1, order: np.ndarray | None = None) -> np.ndarray:
    """Greedy LD pruning (kept for tests/diagnostics; not used by the D6(d) estimator)."""
    p = R.shape[0]
    order = np.arange(p) if order is None else np.asarray(order)
    blocked = np.zeros(p, bool)
    kept = []
    for j in order:
        if blocked[j]:
            continue
        kept.append(j)
        row = np.asarray(R[j], dtype=np.float64)
        blocked |= row * row >= r2_max
        blocked[j] = True
    return np.sort(np.asarray(kept, dtype=np.int64))

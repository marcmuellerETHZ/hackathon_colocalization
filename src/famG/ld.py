"""Batched LD operators on GPU: dense R (B,P,P) or genotype factor X (B,n,P) with R = X^T X / n_ref.

All loci in one batch share the padded width P (bucketed by p, no padding to the global p_max).
Padded columns are all-zero in R / X; callers mask them out of the prior so they carry no mass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .precision import check_ld_dtype


@dataclass
class BatchLD:
    kind: str                    # "dense" | "factor"
    mat: torch.Tensor            # (B,P,P) dense R, or (B,n,P) factor X
    n_ref: torch.Tensor | None   # (B,) float64, factor only (R = X^T X / n_ref)

    @property
    def B(self) -> int:
        return self.mat.shape[0]

    @property
    def P(self) -> int:
        return self.mat.shape[-1]

    @property
    def dtype(self):
        return self.mat.dtype

    def diag(self) -> torch.Tensor:
        """(B,P) float64 diagonal of R."""
        if self.kind == "dense":
            return torch.diagonal(self.mat, dim1=1, dim2=2).to(torch.float64)
        d = torch.empty((self.B, self.P), dtype=torch.float64, device=self.mat.device)
        ch = max(1, int(2 ** 28 // max(1, self.mat.shape[1] * self.P)))   # <= 2 GiB float64 temporaries
        for c0 in range(0, self.B, ch):
            d[c0:c0 + ch] = (self.mat[c0:c0 + ch].to(torch.float64) ** 2).sum(1)
        return d / self.n_ref[:, None]

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """R v for v of shape (B,P) or (B,K,P) (float64 in, float64 out; compute in LD dtype)."""
        squeeze = v.dim() == 2
        vv = v.unsqueeze(1) if squeeze else v          # (B,K,P)
        vc = vv.to(self.mat.dtype)
        if self.kind == "dense":
            out = torch.bmm(vc, self.mat)               # (B,K,P) @ (B,P,P); R symmetric
        else:
            t = torch.bmm(vc, self.mat.transpose(1, 2))  # (B,K,n) = v X^T
            out = torch.bmm(t, self.mat)                # (B,K,P)
            out = out / self.n_ref[:, None, None].to(out.dtype)
        out = out.to(torch.float64)
        return out[:, 0] if squeeze else out

    def corr_sub(self, b: int, idx: torch.Tensor) -> torch.Tensor:
        """|corr| submatrix (m,m) float64 for locus b and variable indices idx (cov2cor of R)."""
        if self.kind == "dense":
            sub = self.mat[b][idx][:, idx].to(torch.float64)
        else:
            Xs = self.mat[b][:, idx].to(torch.float64)
            sub = Xs.T @ Xs / self.n_ref[b]
        dg = torch.sqrt(torch.diagonal(sub))
        return sub / dg[:, None] / dg[None, :]

    def hadamard_quad(self, u: torch.Tensor, chunk_bytes: float = 3e8) -> torch.Tensor:
        """u (B,T,P) float64 -> (B,T) float64 values u_t' (R o R) u_t (R o R = elementwise square of R).
        Factor form: ||X diag(u) X'||_F^2 / n_ref^2."""
        B, T, P = u.shape
        out = torch.empty((B, T), dtype=torch.float64, device=u.device)
        if self.kind == "dense":
            ch = max(1, int(chunk_bytes // (P * P * 8)))
            for c0 in range(0, B, ch):
                R = self.mat[c0:c0 + ch].to(torch.float64, copy=True)   # copy: never square the stored LD in place
                R.mul_(R)
                out[c0:c0 + ch] = (torch.bmm(u[c0:c0 + ch], R) * u[c0:c0 + ch]).sum(-1)
                del R
            return out
        n = self.mat.shape[1]
        ch = max(1, int(chunk_bytes // (n * P * 8 + n * n * 8)))
        for c0 in range(0, B, ch):
            X = self.mat[c0:c0 + ch].to(torch.float64)                      # (c,n,P)
            nr = self.n_ref[c0:c0 + ch]
            for t in range(T):
                M = torch.bmm(X * u[c0:c0 + ch, t, None, :], X.transpose(1, 2))   # (c,n,n)
                out[c0:c0 + ch, t] = (M * M).sum((1, 2)) / (nr * nr)
        return out

    def corr_rows(self, b: int, idx: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
        """rows r0:r1 of the cov2cor(R)[idx, idx] submatrix for locus b (float64)."""
        if self.kind == "dense":
            sub = self.mat[b][idx[r0:r1]][:, idx].to(torch.float64)
            d_all = torch.diagonal(self.mat[b]).to(torch.float64)[idx]
        else:
            X = self.mat[b][:, idx].to(torch.float64)
            sub = X[:, r0:r1].T @ X / self.n_ref[b]
            d_all = (X * X).sum(0) / self.n_ref[b]
        dg = torch.sqrt(d_all)
        return sub / dg[r0:r1, None] / dg[None, :]

    def select(self, idx: torch.Tensor) -> "BatchLD":
        return BatchLD(self.kind, self.mat[idx], None if self.n_ref is None else self.n_ref[idx])


def _as_dev(a, device):
    return a.to(device) if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a), device=device)


def build_batch_ld(lds: list, kind: str, P: int, dtype: torch.dtype, device, symmetrize: bool = False) -> BatchLD:
    """lds: list of numpy arrays (or torch tensors), each (p,p) for dense or (n_ref,p) for factor; padded to width P.
    symmetrize (dense only): store 0.5 (R + R'), formed in float64 (as src/baselines/locus.py:dense_ld)."""
    check_ld_dtype(dtype)
    B = len(lds)
    if kind == "dense":
        out = torch.zeros((B, P, P), dtype=dtype, device=device)
        for i, R in enumerate(lds):
            p = R.shape[0]
            Rt = _as_dev(R, device)
            if symmetrize:
                Rt = Rt.to(torch.float64)
                Rt = 0.5 * (Rt + Rt.T)
            out[i, :p, :p] = Rt.to(dtype)
        return BatchLD("dense", out, None)
    nmax = max(X.shape[0] for X in lds)
    out = torch.zeros((B, nmax, P), dtype=dtype, device=device)
    n_ref = torch.zeros(B, dtype=torch.float64, device=device)
    for i, X in enumerate(lds):
        n, p = X.shape
        out[i, :n, :p] = _as_dev(X, device).to(dtype)
        n_ref[i] = n
    return BatchLD("factor", out, n_ref)

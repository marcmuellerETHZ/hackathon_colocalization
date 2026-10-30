"""NeuralColoc network (DESIGN v4 §5 C "Architecture").

Two streams, one readout:
  * screen stream: transformer over the M screened variants (tokens = allele-canonicalised per-variant features +
    a projection of the global features), multi-head attention with an additive LD bias
    b_h(i, j) = MLP_h(r_ij, r_ij², |r_ij|) of the canonicalised inference LD; no positional encoding
    (permutation-equivariant over tokens);
  * all-variant stream (README decision C-A2): a per-variant coefficient MLP over every window variant, without
    attention (linear in p), so variants outside the screen still enter the readout.
Readout ("BF-structured pooling", README decision C-A1): every stream emits K per-variant energies that are
quadratic in allele-invariant z statistics,
        e_k(j) = a_k(h_j) + sum_m c_km(h_j) q_m(j),   q = (z1², z2², z1 z2, zc1², zc2², zc1 zc2) / 2,
(the form of a Gaussian single-variant log Bayes factor), pooled by log-sum-exp over variants (a permutation-
invariant pooled readout). Class logits = A E + b + MLP([tanh(C E / 8), pooled token embeddings, globals]),
E = [E_screen; E_all] in R^{2K}. Output: 5 logits (C0: coloc's H0..H4; main task: oracle classes S, D, W, G, U),
plus a per-token shared-PIP head (softmax over screened tokens + one 'none' slot).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from famC.features import N_ALL_FEAT, N_GLOB_FEAT, N_QUAD, N_TOK_FEAT


class LDBias(nn.Module):
    def __init__(self, H: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, H))

    def forward(self, R):                              # R [B, M, M]
        x = torch.stack([R, R * R, R.abs()], -1)
        return self.net(x).permute(0, 3, 1, 2)          # [B, H, M, M]


class Block(nn.Module):
    def __init__(self, d: int, H: int, ff: int = 4, dropout: float = 0.0):
        super().__init__()
        self.H = H
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d))
        self.dropout = dropout

    def forward(self, x, bias):                         # bias [B, H, M, M] incl. -inf key mask
        B, M, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, M, 3, self.H, d // self.H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(B, M, d))
        return x + self.ff(self.ln2(x))


def mlp(i, h, o, n=2):
    layers = [nn.Linear(i, h), nn.GELU()]
    for _ in range(n - 1):
        layers += [nn.Linear(h, h), nn.GELU()]
    layers.append(nn.Linear(h, o))
    return nn.Sequential(*layers)


DIAG = (0, 1, 3, 4)      # q terms that are squares (z1², z2², zc1², zc2²)
CROSS = (2, 5)           # cross terms (z1 z2, zc1 zc2)


class EnergyHead(nn.Module):
    """Per-variant K energies quadratic in the q basis; coefficients from a per-variant embedding.

    param = "free": every coefficient is a linear output of the embedding.
    param = "sat" (README C-A5): square-term coefficients c = kappa (1 - exp(v)), v linear in the embedding,
    kappa = 0.5 exp(k) a learned scalar per (energy, term). A Gaussian single-variant log BF has c = 0.5 s²/(1+s²)
    -> 0.5 as the prior-to-noise ratio s grows; here that limit is reached through exp(v) -> 0, which keeps
    the coefficient error small in absolute terms at large |z| (c z² must be accurate to << 1)."""

    def __init__(self, d: int, K: int, param: str = "free", init_scale: float = 0.1):
        super().__init__()
        self.K, self.param = K, param
        self.coef = nn.Linear(d, K * (1 + N_QUAD))
        nn.init.zeros_(self.coef.weight)
        with torch.no_grad():
            b = torch.zeros(K, 1 + N_QUAD)
            c0 = init_scale * torch.rand(K, N_QUAD)
            b[:, 1:] = c0
            if param == "sat":
                b[:, 1 + torch.tensor(DIAG)] = torch.log(1 - 2 * c0[:, DIAG])
            self.coef.bias.copy_(b.reshape(-1))
        if param == "sat":
            self.logk = nn.Parameter(torch.zeros(K, len(DIAG)))

    def forward(self, h, q, mask):                      # h [B,N,d], q [B,N,6], mask [B,N]
        c = self.coef(h).reshape(*h.shape[:2], self.K, 1 + N_QUAD)
        a, c = c[..., 0], c[..., 1:]
        if self.param == "sat":
            kap = 0.5 * torch.exp(self.logk)
            cd = kap * (1 - torch.exp(c[..., DIAG].clamp(max=20.0)))
            e = a + (cd * q[:, :, None, DIAG]).sum(-1) + (c[..., CROSS] * q[:, :, None, CROSS]).sum(-1)
        else:
            e = a + (c * q[:, :, None, :]).sum(-1)                        # [B, N, K]
        e = e.masked_fill(~mask[..., None], -math.inf)
        return e


class NeuralColoc(nn.Module):
    def __init__(self, d: int = 128, layers: int = 4, H: int = 4, K: int = 16, n_out: int = 5, d_all: int = 64,
                 use_ld: bool = True, use_all: bool = True, dropout: float = 0.0, energy: str = "free",
                 pip_init: float = 0.1):
        super().__init__()
        self.use_ld, self.use_all, self.K, self.n_out = use_ld, use_all, K, n_out
        self.tok_in = nn.Linear(N_TOK_FEAT, d)
        self.glob_in = mlp(N_GLOB_FEAT, d, d, 1)
        self.ldb = LDBias(H) if use_ld else None
        self.blocks = nn.ModuleList([Block(d, H, dropout=dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.e_tok = EnergyHead(d, K, energy)
        if use_all:
            self.all_net = mlp(N_ALL_FEAT + N_GLOB_FEAT, d_all, d_all, 2)
            self.e_all = EnergyHead(d_all, K, energy)
        nE = K * (2 if use_all else 1)
        self.lin = nn.Linear(nE, n_out)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        self.cmix = nn.Linear(nE, 32)
        self.pool_q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.corr = mlp(32 + 2 * d + N_GLOB_FEAT, 2 * d, n_out, 2)
        # zero-initialised quadratic terms: PIP logits start flat (z² terms of O(10^3) would otherwise dominate
        # the gradient of the whole network at initialisation)
        self.pip = EnergyHead(d, 1, energy, init_scale=pip_init)
        self.pip_null = mlp(2 * d + N_GLOB_FEAT, d, 1, 1)

    def forward(self, f: dict) -> dict:
        tok, valid, glob = f["tok"], f["valid"], f["glob"]
        B, M, _ = tok.shape
        g = self.glob_in(glob)
        x = self.tok_in(tok) + g[:, None, :]
        keymask = torch.zeros(B, 1, 1, M, device=tok.device, dtype=x.dtype).masked_fill(~valid[:, None, None, :], -1e9)
        if self.use_ld:
            bias = self.ldb(f["Rmm"]) + keymask
        else:
            bias = keymask.expand(B, self.blocks[0].H, M, M)
        for blk in self.blocks:
            x = blk(x, bias)
        h = self.ln_f(x)
        e_tok = self.e_tok(h, f["q_tok"], valid)                 # [B, M, K]
        E = [torch.logsumexp(e_tok, 1)]                          # [B, K]
        if self.use_all:
            mask = f["mask"]
            P = mask.shape[1]
            a_in = torch.cat([f["all_feat"], glob[:, None, :].expand(B, P, glob.shape[-1])], -1)
            ha = self.all_net(a_in)
            e_all = self.e_all(ha, f["q_all"], mask)
            E.append(torch.logsumexp(e_all, 1))
        E = torch.cat(E, -1)                                      # [B, nE]
        vm = valid[..., None].to(h.dtype)
        hmean = (h * vm).sum(1) / vm.sum(1).clamp_min(1)
        att = (h @ self.pool_q).masked_fill(~valid, -1e9).softmax(-1)
        hatt = (att[..., None] * h).sum(1)
        hp = torch.cat([hmean, hatt], -1)
        logits = self.lin(E) + self.corr(torch.cat([torch.tanh(self.cmix(E) / 8), hp, glob], -1))
        pip_tok = self.pip(h, f["q_tok"], valid)[..., 0]          # [B, M]
        pip_null = self.pip_null(torch.cat([hp, glob], -1))       # [B, 1]
        pip_logits = torch.cat([pip_tok, pip_null], -1)
        return dict(logits=logits, pip_logits=pip_logits, E=E)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

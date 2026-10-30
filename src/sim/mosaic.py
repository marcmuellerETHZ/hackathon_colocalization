"""Li-Stephens haplotype-mosaic synthetic populations (DESIGN §2.1, LD world 1).

A mosaic population is a FIXED genome-wide object, a pure function of (seed, key, pool, N_e, theta):

* each synthetic haplotype h (h = 0..2N-1; individual i = haplotypes 2i, 2i+1) copies source haplotypes of
  the pool (K = 2 x pool size) along each chromosome;
* copying switches follow a Poisson process in genetic distance with intensity 4 N_e / K per Morgan
  (Li & Stephens 2003 transition); after a switch the new source is uniform over the K haplotypes;
  the switch process of (haplotype, chromosome) is drawn once from rng(seed, "mosaic_v1", key, chrom);
* the source copied at a variant with genetic position g is source number #(switches <= g), so the result
  at a variant does not depend on which window it is materialised in;
* miscopy: each copied allele is flipped with probability theta; the flipped haplotypes at a variant are
  drawn from rng(seed, "mosaic_v1", key, chrom, "miscopy", pos), again independent of the window.

Nothing is stored on disk: windows are materialised on the GPU on the fly (README D1). Variants are the
target-pool QC set (src/sim/world.py::window_variants) for every pool, so all mosaics share coordinates.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from common.seeds import rng
from sim import params
from sim.world import chrom_cm_length, chrom_name, pools, source_haplotypes

_CHUNK = 4096


class Mosaic:
    def __init__(self, pool: str, n_ind: int, key: str, ne: float = params.NE, theta: float = params.THETA,
                 seed: int = params.MOSAIC_SEED):
        self.pool, self.n_ind, self.key = pool, int(n_ind), key
        self.ne, self.theta, self.seed = float(ne), float(theta), int(seed)
        self.K = 2 * len(pools()[pool])
        self.H = 2 * self.n_ind
        self._sw_cache: dict[str, tuple] = {}

    @classmethod
    def named(cls, name: str, **kw) -> "Mosaic":
        pool, n, key = params.POPULATIONS[name]
        kw.setdefault("ne", params.POP_NE.get(name, params.NE))
        return _named(name, pool, n, key, tuple(sorted(kw.items())))

    def __repr__(self):
        return f"Mosaic(pool={self.pool}, N={self.n_ind}, key={self.key}, Ne={self.ne}, theta={self.theta})"

    # ------------------------------------------------------------------ switch process
    def switches(self, chrom: str):
        """(offs[H+1], s_cm[E] sorted within haplotype, src[E+H], L_cm) for one chromosome."""
        chrom = chrom_name(chrom)
        if chrom not in self._sw_cache:
            g = rng(self.seed, "mosaic_v1", self.key, self.pool, chrom, "switch")
            L = chrom_cm_length(chrom) + 1.0
            rate = 4.0 * self.ne / (self.K * 100.0)              # switches per cM per haplotype
            n_h = g.poisson(rate * L, size=self.H)
            E = int(n_h.sum())
            s = g.uniform(0.0, L, size=E)
            hap = np.repeat(np.arange(self.H), n_h)
            order = np.lexsort((s, hap))
            s = s[order]
            src = g.integers(0, self.K, size=E + self.H, dtype=np.int32)
            offs = np.concatenate([[0], np.cumsum(n_h)]).astype(np.int64)
            if len(self._sw_cache) >= 3:
                self._sw_cache.pop(next(iter(self._sw_cache)))
            self._sw_cache[chrom] = (offs, s, src, L)
        return self._sw_cache[chrom]

    def _source_index(self, chrom: str, cm: np.ndarray):
        """Per haplotype: base offset into src (int64 [H]) and in-window switch events (h_e, j_e)."""
        offs, s, src, L = self.switches(chrom)
        H = self.H
        hap_of = np.repeat(np.arange(H), np.diff(offs))
        keyed = hap_of * (L + 1.0) + s                             # globally sorted
        g0, g1 = float(cm[0]), float(cm[-1])
        k0 = np.searchsorted(keyed, np.arange(H) * (L + 1.0) + g0, side="left") - offs[:-1]
        base = offs[:-1] + np.arange(H) + k0                       # index into src of the source at g0
        inwin = (s >= g0) & (s <= g1)
        h_e = hap_of[inwin]
        j_e = np.searchsorted(cm, s[inwin], side="left")
        return base, h_e, j_e, src

    def miscopy(self, chrom: str, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(haplotype, variant column) pairs whose allele is flipped."""
        hs, js = [], []
        for j, p_ in enumerate(np.asarray(pos)):
            g = rng(self.seed, "mosaic_v1", self.key, self.pool, chrom_name(chrom), "miscopy", int(p_))
            n = g.binomial(self.H, self.theta)
            if n:
                hs.append(g.integers(0, self.H, size=n))
                js.append(np.full(n, j))
        if not hs:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(hs).astype(np.int64), np.concatenate(js).astype(np.int64)

    # ------------------------------------------------------------------ materialisation
    @torch.no_grad()
    def haplotypes_from_source(self, chrom: str, Hsrc: np.ndarray, v: pd.DataFrame,
                               device: str | torch.device = "cuda") -> torch.Tensor:
        """Synthetic haplotypes uint8 [2N, p] on `device`, given source haplotypes of this pool at v."""
        chrom = chrom_name(chrom)
        assert Hsrc.shape == (self.K, len(v)), (Hsrc.shape, self.K, len(v))
        p = len(v)
        out = torch.empty((self.H, p), dtype=torch.uint8, device=device)
        if p == 0:
            return out
        cm = v["cm"].to_numpy(np.float64)
        assert np.all(np.diff(cm) >= 0)
        base, h_e, j_e, src = self._source_index(chrom, cm)
        src_t = torch.as_tensor(src, device=device, dtype=torch.int64)
        hs_t = torch.as_tensor(np.ascontiguousarray(Hsrc), device=device, dtype=torch.uint8).reshape(-1)
        cols = torch.arange(p, device=device, dtype=torch.int64)
        for a in range(0, self.H, _CHUNK):
            b = min(self.H, a + _CHUNK)
            sel = (h_e >= a) & (h_e < b)
            cnt = torch.zeros((b - a, p + 1), dtype=torch.int32, device=device)
            if sel.any():
                hh = torch.as_tensor(h_e[sel] - a, device=device, dtype=torch.int64)
                jj = torch.as_tensor(j_e[sel], device=device, dtype=torch.int64)
                cnt.index_put_((hh, jj), torch.ones(len(hh), dtype=torch.int32, device=device), accumulate=True)
            seg = torch.cumsum(cnt[:, :p], dim=1, dtype=torch.int64)
            idx = src_t[torch.as_tensor(base[a:b], device=device)[:, None] + seg]     # source hap [h, j]
            out[a:b] = hs_t[idx * p + cols[None, :]]
            del cnt, seg, idx
        fh, fj = self.miscopy(chrom, v["pos"].to_numpy())
        if len(fh):
            fh_t = torch.as_tensor(fh, device=device)
            fj_t = torch.as_tensor(fj, device=device)
            # duplicates (same hap drawn twice at a variant) flip once: use assignment, not xor-accumulate
            out[fh_t, fj_t] = 1 - out[fh_t, fj_t]
        return out

    def haplotypes(self, chrom, start: int, end: int, device="cuda") -> tuple[torch.Tensor, pd.DataFrame]:
        Hsrc, v = source_haplotypes(self.pool, chrom, start, end)
        return self.haplotypes_from_source(chrom_name(chrom), Hsrc, v, device), v

    def dosages(self, chrom, start: int, end: int, device="cuda") -> tuple[torch.Tensor, pd.DataFrame]:
        """ALT dosages uint8 [N, p] (individual i = haplotypes 2i, 2i+1)."""
        h, v = self.haplotypes(chrom, start, end, device)
        return h[0::2] + h[1::2], v


@lru_cache(maxsize=8)
def _named(name, pool, n, key, kw):
    return Mosaic(pool, n, key, **dict(kw))


def hap_to_dosage(h: torch.Tensor) -> torch.Tensor:
    return h[0::2] + h[1::2]

"""Bank of real 1000G LD windows on the GPU (Family C, C0 world and adapters).

A window is a clipped gene window of `datasets/splits/genes_v1.tsv.gz` (via `sim.windows.split_windows`, which refuses
the test split). Its variant set is the simulation world's analysis set (QC'd SNVs, pool-A MAF >= 0.01,
`sim.world.window_variants` rule); genotypes are the real pool-A individuals (n = 254) from
`data.genotypes.load_genotypes`. LD is R = X'X / n with X the standardized dosages, so the bank stores only
int8 dosages + per-variant mean/sd and materialises standardized X per batch.

Windows are kept sorted by p, so a batch drawn from a contiguous run of the sorted list needs little padding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from common.seeds import rng as seeded_rng

FAMC_TAG = "famC"
MIN_P = 200


@dataclass
class WindowInfo:
    chrom: str
    start: int
    end: int
    p: int
    gene_id: str
    block_id: str


class LDBank:
    """Real pool-A genotype windows of one split, on `device`.

    Attributes
    ----------
    G    : list of uint8 tensors [n, p_w]  (ALT dosages)
    mean, sd : list of float32 tensors [p_w]
    af   : list of float64 numpy arrays [p_w] (in-sample ALT frequency of the pool)
    info : list[WindowInfo]   (sorted by p)
    """

    def __init__(self, split: str, n_windows: int, seed: int, pool: str = "A", device="cuda", min_maf: float = 0.01,
                 verbose: bool = False):
        from data.genotypes import load_genotypes
        from sim.windows import split_windows
        from sim.world import pools

        if split == "test":
            raise PermissionError("Family C never loads test-split windows (DESIGN §7.3)")
        self.split, self.pool, self.device = split, pool, torch.device(device)
        w = split_windows(split).drop_duplicates(["chr", "window_start", "window_end"]).reset_index(drop=True)
        g = seeded_rng(seed, FAMC_TAG, "ldbank", split, pool)
        order = g.permutation(len(w))[: min(n_windows, len(w))]
        # load chromosome by chromosome (the variant-table cache holds few chromosomes); final order is by p
        order = sorted(order.tolist(), key=lambda i: (str(w.iloc[i]["chr"]), int(w.iloc[i]["window_start"])))
        samples = pools()[pool]
        self.n = len(samples)
        G_list, info, af_list = [], [], []
        for k, i in enumerate(order):
            r = w.iloc[int(i)]
            G, v = load_genotypes(r["chr"], int(r["window_start"]), int(r["window_end"]), samples,
                                  maf_pool="A", min_maf=min_maf)
            if G.shape[1] < MIN_P:
                continue
            Gt = torch.as_tensor(G.astype(np.uint8), device=self.device)
            f = G.astype(np.float64).mean(0) / 2.0
            # non-constant dosage in the pool (needed for standardisation; e.g. all-heterozygous artefacts exist)
            keep = (np.minimum(f, 1 - f) > 0) & (G.std(0) > 0)
            if not keep.all():
                Gt = Gt[:, torch.as_tensor(np.where(keep)[0], device=self.device)]
                f = f[keep]
            G_list.append(Gt)
            af_list.append(f)
            info.append(WindowInfo(r["chr"], int(r["window_start"]), int(r["window_end"]), int(Gt.shape[1]),
                                   str(r["gene_id"]), str(r["block_id"])))
            if verbose and k % 200 == 0:
                print(f"[ldbank {split}] {k}/{len(order)} windows", flush=True)
        srt = np.argsort([x.p for x in info], kind="stable")
        self.G = [G_list[i] for i in srt]
        self.af = [af_list[i] for i in srt]
        self.info = [info[i] for i in srt]
        self.p = np.array([x.p for x in self.info])
        self.mean, self.sd = [], []
        for Gt in self.G:
            x = Gt.float()
            m = x.mean(0)
            s = x.std(0, unbiased=False)
            self.mean.append(m)
            self.sd.append(s)
        assert all(float(s.min()) > 0 for s in self.sd)

    def __len__(self):
        return len(self.G)

    def nbytes(self) -> int:
        return int(sum(x.numel() for x in self.G))

    def standardized(self, widx) -> tuple[torch.Tensor, torch.Tensor]:
        """X [Wb, n, P] float32 standardized (zero in padding), mask [Wb, P] bool."""
        widx = list(int(i) for i in widx)
        P = int(max(self.p[i] for i in widx))
        X = torch.zeros(len(widx), self.n, P, device=self.device)
        mask = torch.zeros(len(widx), P, dtype=torch.bool, device=self.device)
        for b, i in enumerate(widx):
            p = int(self.p[i])
            X[b, :, :p] = (self.G[i].float() - self.mean[i]) / self.sd[i]
            mask[b, :p] = True
        return X, mask

    def af_padded(self, widx, fill: float = 0.5) -> torch.Tensor:
        widx = list(int(i) for i in widx)
        P = int(max(self.p[i] for i in widx))
        out = torch.full((len(widx), P), fill, dtype=torch.float64, device=self.device)
        for b, i in enumerate(widx):
            out[b, : self.p[i]] = torch.as_tensor(self.af[i], device=self.device)
        return out

    def sample_windows(self, g: torch.Generator, n: int) -> np.ndarray:
        """n windows from a contiguous run of the p-sorted list (start uniform), shuffled."""
        W = len(self)
        n = min(n, W)
        s = int(torch.randint(0, W - n + 1, (1,), generator=g, device="cpu").item())
        return np.arange(s, s + n)

    def all_batches(self, n: int):
        """Deterministic sweep over the bank in p-sorted chunks of n windows (evaluation)."""
        for s in range(0, len(self), n):
            yield np.arange(s, min(s + n, len(self)))


def torch_gen(seed: int, *keys, device="cpu") -> torch.Generator:
    """torch.Generator seeded from the fixed seed list via common.seeds.derive."""
    from common.seeds import derive
    g = torch.Generator(device=device)
    g.manual_seed(derive(seed, FAMC_TAG, *keys) % (2 ** 63 - 1))
    return g


def logu(g: torch.Generator, lo: float, hi: float, size, device) -> torch.Tensor:
    u = torch.rand(size, generator=g, device=g.device, dtype=torch.float64).to(device)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))

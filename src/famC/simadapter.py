"""Adapter from the simulation agent's on-the-fly TRAIN-split API (`sim.train_api.TrainSampler`) to Family C's
two-stage features, plus sharded storage of training/validation records.

`FamCSampler` subclasses `TrainSampler` and replaces only its per-locus step `_locus`: the window loop, the cell
prior (`draw_cell`: levels L0-L4 with L6 held out, pairing types with their overlap grids, both rho signs), the
seeds (rng(seed, "train", step, k), DESIGN §9) and the GPU window contexts are the simulation agent's. Per locus we
call the same generator functions as TrainSampler (draw_candidate, observed_stats, ld_source, rho_hat_for) and
keep what Family C needs beyond the screened view: the all-variant stream (z, se, MAF, N, LD rows of the leads)
and the screen of `famC.features.extract` (identical to famB's screen without collapse).

Class balancing (README decision C-T2): candidates are kept with probability ACCEPT[class] (decided from the
oracle class before the noise draw, so it costs only the candidate); the network's logits are corrected by
-log ACCEPT at inference so outputs refer to the generator's natural mixture.

Validation records come from the same code on VALIDATION-split windows with seed 20261029 (DESIGN §9 "NPE
tuning / validation draws"); `ValidSampler` lifts TrainSampler's train-only guard for that purpose only.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from famC.features import extract

CLASS_ORDER = ("S", "D", "W", "G", "U")          # Family C output order
TYPE_CODE = {"gwas": 0.0, "mol": 1.0}
ACCEPT = {"S": 1.0, "D": 1.0, "U": 1.0, "G": 0.4, "W": 0.12}
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "famC"


class LDSourceOp:
    """sim.generator.LDSource (one locus) as a famC LD access object with B = 1."""

    def __init__(self, src):
        self.src = src

    def rows(self, idx):
        return self.src.rows(idx[0].cpu().numpy())[None].double()

    def sub(self, idx):
        return self.src.block(idx[0].cpu().numpy())[None].double()


def _import_sim():
    from sim import train_api
    from sim.generator import draw_candidate, ld_source, observed_stats, rho_hat_for
    return train_api, draw_candidate, ld_source, observed_stats, rho_hat_for


def locus_record(ctx, cell, g: np.random.Generator, M: int, accept: dict | None, level_names, pairing_names,
                 n_noise: int = 1):
    """One candidate -> (record, meta) [n_noise = 1] or a list of n_noise (record, meta) pairs, each with its own
    noise draw (observed statistics, LD at the level, rho-hat) for the same architecture and study samples
    (README C-T6); None if there is no candidate, "rejected" if the class was not accepted."""
    _, draw_candidate, ld_source, observed_stats, rho_hat_for = _import_sim()
    c = draw_candidate(ctx, cell, g)
    if c is None:
        return None
    o = c.oracle
    if accept is not None and g.random() > accept[o.cls]:
        return "rejected"
    out = [_observe(ctx, cell, c, g, M, level_names, pairing_names) for _ in range(n_noise)]
    return out[0] if n_noise == 1 else out


def _observe(ctx, cell, c, g, M, level_names, pairing_names):
    _, draw_candidate, ld_source, observed_stats, rho_hat_for = _import_sim()
    o = c.oracle
    ob = observed_stats(ctx, c, g)
    keep = ob["keep"]
    src = ld_source(ctx, c, keep)
    rh = rho_hat_for(c.ps, ob["z1"], ob["z2"], src, g, cell.level)   # level as sim.train_api (D14 residualisation)
    dev = ctx.device
    P = len(keep)
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), device=dev)[None]
    f1, f2 = c.s1.af[keep], c.s2.af[keep]
    rho_hat = float(rh["rho_hat"]) if np.isfinite(rh["rho_hat"]) else 0.0
    rho_se = float(rh["rho_se"]) if np.isfinite(rh["rho_se"]) else 1.0
    b = dict(z1=t(ob["z1"]), z2=t(ob["z2"]), se1=t(ob["se1"]), se2=t(ob["se2"]),
             n1=t(np.full(P, float(c.ps.t1.N))), n2=t(np.full(P, float(c.ps.t2.N))),
             maf1=t(np.minimum(f1, 1 - f1)), maf2=t(np.minimum(f2, 1 - f2)), maf_ref=t(src.maf_ref),
             mask=torch.ones(1, P, dtype=torch.bool, device=dev),
             n_ref=torch.tensor([float(src.n_ref_attr)], dtype=torch.float64, device=dev),
             rho_hat=torch.tensor([rho_hat], dtype=torch.float64, device=dev),
             rho_se=torch.tensor([rho_se], dtype=torch.float64, device=dev),
             type1=torch.tensor([TYPE_CODE[c.ps.t1.kind]], device=dev),
             type2=torch.tensor([TYPE_CODE[c.ps.t2.kind]], device=dev))
    ld = LDSourceOp(src)
    rec = extract(b, ld, M=M)
    # shared-variant target token (S only): the class-determining shared variant if screened, else the screened
    # token with the largest r² (inference LD) to it if r² >= 0.8, else 'none' (= M)
    pip_t = M
    if o.cls == "S" and o.cd1 is not None:
        pos = np.searchsorted(keep, int(o.cd1))
        if pos < len(keep) and keep[pos] == int(o.cd1):
            sel = rec["sel"][0]
            hit = (sel == pos) & rec["valid"][0]
            if bool(hit.any()):
                pip_t = int(torch.nonzero(hit)[0, 0])
            else:
                r = ld.rows(torch.tensor([[pos]], device=dev))[0, 0]
                r2 = (r[sel] ** 2).masked_fill(~rec["valid"][0], -1)
                k = int(r2.argmax())
                if float(r2[k]) >= 0.8:
                    pip_t = k
    # the network input is rec (built from b, which holds rho_hat / rho_se only); the true rho c.ps.rho goes to
    # meta (labels / monitoring), never to the record
    meta = torch.tensor([CLASS_ORDER.index(o.cls), o.mu1, o.mu2, o.r2 if np.isfinite(o.r2) else -1.0,
                         o.d if np.isfinite(o.d) else -1.0, c.ps.rho, level_names.index(cell.level),
                         pairing_names.index(cell.pairing), cell.overlap, P, c.ps.t1.N, c.ps.t2.N, rho_hat, pip_t],
                        dtype=torch.float64)
    return compact(rec), meta


META_COLS = ("y", "mu1", "mu2", "r2", "d", "rho", "level", "pairing", "overlap", "p", "N1", "N2", "rho_hat", "pip_t")
MI_META = {c: i for i, c in enumerate(META_COLS)}


def compact(rec: dict) -> dict:
    """GPU record (B = 1) -> CPU tensors; z columns fp32, other per-variant columns fp16 (README C-T3)."""
    v = rec["valid"][0]
    m = rec["mask"][0]
    P = int(m.sum())
    out = {
        "tok_raw": rec["tok_raw"][0].float().cpu(),
        "src": rec["src"][0].to(torch.int8).cpu(),
        "valid": v.cpu(),
        "Rmm": rec["Rmm"][0].half().cpu() if rec["Rmm"] is not None else None,
        "all_z": rec["all_raw"][0, :P, :4].float().cpu(),
        "all_o": rec["all_raw"][0, :P, 4:].half().cpu(),
        "glob_raw": rec["glob_raw"][0].float().cpu(),
        "sel": rec["sel"][0].to(torch.int32).cpu(),
        "lead1": int(rec["lead1"][0]), "lead2": int(rec["lead2"][0]),
    }
    return out


def collate_records(recs: list[dict], device="cuda") -> dict:
    """List of compact records -> batched extract()-style record on `device` (padding P to the batch max)."""
    B = len(recs)
    M = recs[0]["tok_raw"].shape[0]
    P = max(r["all_z"].shape[0] for r in recs)
    all_raw = torch.zeros(B, P, 13, dtype=torch.float32)
    mask = torch.zeros(B, P, dtype=torch.bool)
    for i, r in enumerate(recs):
        p = r["all_z"].shape[0]
        all_raw[i, :p, :4] = r["all_z"]
        all_raw[i, :p, 4:] = r["all_o"].float()
        mask[i, :p] = True
    has_R = recs[0]["Rmm"] is not None
    out = dict(
        tok_raw=torch.stack([r["tok_raw"] for r in recs]).to(device),
        src=torch.stack([r["src"] for r in recs]).long().to(device),
        valid=torch.stack([r["valid"] for r in recs]).to(device),
        Rmm=torch.stack([r["Rmm"] for r in recs]).float().to(device) if has_R else None,
        all_raw=all_raw.to(device), mask=mask.to(device),
        glob_raw=torch.stack([r["glob_raw"] for r in recs]).to(device),
        sel=torch.stack([r["sel"] for r in recs]).long().to(device),
        lead1=torch.tensor([r["lead1"] for r in recs], device=device),
        lead2=torch.tensor([r["lead2"] for r in recs], device=device),
    )
    return out


def make_sampler_class():
    train_api = _import_sim()[0]

    class FamCSampler(train_api.TrainSampler):
        """TrainSampler whose per-locus step yields Family C records (see module docstring)."""

        def __init__(self, *a, accept: dict | None = ACCEPT, n_noise: int = 1, **kw):
            super().__init__(*a, **kw)
            self.accept, self.n_noise = accept, n_noise
            self.stats.setdefault("rejected", 0)

        def _locus(self, ctx, cell, g):
            r = locus_record(ctx, cell, g, self.M, self.accept, train_api.LEVELS, train_api.PAIRINGS, self.n_noise)
            if isinstance(r, str):
                self.stats["rejected"] += 1
                return None
            return r

    return FamCSampler


def make_valid_sampler_class():
    """Same sampler on VALIDATION windows (seed 20261029). TrainSampler only guards split == 'train'; the window
    table is replaced after construction. Test windows are refused by sim.windows.split_windows itself."""
    from sim.windows import split_windows
    Base = make_sampler_class()

    class ValidSampler(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.wins = split_windows("valid")

    return ValidSampler


def write_shards(sampler, out_dir: Path, n_loci: int, shard_size: int = 2048, tag: str = "", log_every: int = 1):
    """Draw window visits until n_loci records are written, in shards of shard_size (torch.save)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buf, metas = [], []
    n, shard = 0, 0
    t0 = time.time()
    while n < n_loci:
        items = sampler.window_loci(sampler.step)
        sampler.step += 1
        for it in items:
            for rec, meta in (it if isinstance(it, list) else [it]):
                buf.append(rec)
                metas.append(meta)
        while buf and n < n_loci and (len(buf) >= shard_size or n + len(buf) >= n_loci):
            k = min(shard_size, len(buf), n_loci - n)
            torch.save({"recs": buf[:k], "meta": torch.stack(metas[:k]), "meta_cols": META_COLS,
                        "step_end": sampler.step},
                       out_dir / f"shard{tag}_{shard:05d}.pt")
            buf, metas = buf[k:], metas[k:]
            n += k
            shard += 1
            el = time.time() - t0
            if shard % log_every == 0:
                print(f"[{tag}] shard {shard} loci {n} {n / el:.1f} loci/s stats {sampler.stats}", flush=True)
    return n

"""C1 kill-test training (DESIGN v4 §5 C): in world L0* (Family B's simulator-prior world, where B's oracle-class
posterior is Bayes-optimal), NeuralColoc must match B's oracle-class posteriors, expected KL(B || C) < 0.02 under
the world's natural class distribution (DECISIONS P3-4), reported as a learning curve over training-set size.

Training data (README C1): `sim.l0star` (the simulation agent's L0* generator) on TRAIN-split windows only, with
exactly the generator settings of `sim.l0star.build_set` / `sim.l0star.main` (the l0star_v2 spec):
  * window visit: a train window chosen uniformly at random; windows with < 200 variants are skipped (as build_set);
  * segment: `make_segment(ctx, g)` with its defaults (p = P_DEFAULT = 20 SNPs, 100 kb, MAF >= 0.05, NFE mosaic LD,
    R~ = (1 - eta) R + eta I, eta = ETA = 1e-3); several segments per visit (same marginal segment distribution);
  * locus: `sample_locus(seg, g, rho, kinds=("gwas", "mol"))` with rho uniform over `sim.l0star.L0STAR_RHO`
    (0, +-0.3, +-0.5, +-0.8); the natural class mixture (no rejection);
  * inputs exactly as build_set writes them: z, MAF, nominal N (100k GWAS-like, 1k molecular),
    se = 1/sqrt(2 N f (1 - f)), dense LD = R~ (float32), n_ref = 1e9, rho_hat = rho, rho_se = 0 (in L0* sets the
    methods are given the true rho as rho_hat by the world's definition, sim.l0star docstring), trait types
    GWAS x molecular. tests/famC/test_c1_stream.py checks that a stream batch gives the same features as the
    method path (`famC.method`) on the LocusData that build_set would write for the same locus.
Labels: hard oracle classes from `sample_locus` (sim's oracle). Loss: unweighted cross-entropy on the natural
stream, so the network's softmax estimates the natural-mixture posterior directly; the checkpoint config records
`logit_correction = "none"` (the main-task models instead use class-weighted CE and `train_main.natural_logits`).

Evaluation against B needs truth (class shares, oracle labels) and therefore lives OUTSIDE the method package
(DECISIONS P3-21): `scripts/famC_eval/c1_eval.py`. This module only exposes `predict_l0star`, which never reads
truth.

    python -m famC.c1 train --run c1_v2 --n-train 16000000 --eval-at 1000000,2000000,4000000,8000000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from common.seeds import REPLICATE_SEED, rng
from famC.features import DenseLD, build, extract
from famC.model import NeuralColoc, n_params

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "famC"
L0S_CLASSES = ("S", "D", "W", "G", "U")      # = famC.simadapter.CLASS_ORDER
N_NOMINAL = (100_000, 1_000)                 # = sim.l0star.build_set n_nominal
KINDS = ("gwas", "mol")                      # = sim.l0star.build_set kinds
MIN_WINDOW_P = 200                           # = sim.l0star.build_set: segment only if ctx.p >= 200


def l0star_batch(segs, loci, device="cuda") -> tuple[dict, DenseLD]:
    """Stack L0* loci (all with p SNPs) into a famC batch dict + dense LD, with the inputs an L0* locus set gives
    a method. The true-rho field of a locus enters only as rho_hat, which is what L0* sets give methods."""
    B = len(loci)
    p = len(segs[0].maf)
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)
    maf = t(np.stack([s.maf for s in segs]))
    v = 2 * maf * (1 - maf)
    N1, N2 = N_NOMINAL
    b = dict(z1=t(np.stack([x["z1"] for x in loci])), z2=t(np.stack([x["z2"] for x in loci])),
             se1=1 / torch.sqrt(N1 * v), se2=1 / torch.sqrt(N2 * v),
             n1=torch.full((B, p), float(N1), dtype=torch.float64, device=device),
             n2=torch.full((B, p), float(N2), dtype=torch.float64, device=device),
             maf1=maf, maf2=maf, maf_ref=maf, mask=torch.ones(B, p, dtype=torch.bool, device=device),
             n_ref=torch.full((B,), 1e9, dtype=torch.float64, device=device),
             rho_hat=t([x["rho"] for x in loci]), rho_se=torch.zeros(B, dtype=torch.float64, device=device),
             type1=torch.zeros(B, device=device), type2=torch.ones(B, device=device))
    # R~ exactly as stored in L0* sets (build_set writes seg.Rt.astype(float32))
    R = torch.as_tensor(np.stack([s.Rt.astype(np.float32) for s in segs]), dtype=torch.float32, device=device)
    return b, DenseLD(R)


class L0StarStream:
    """Train-split L0* loci: window visit k -> up to `segs_per_window` segments x `per_seg` loci."""

    def __init__(self, seed_key: str, per_seg: int = 128, segs_per_window: int = 32, device="cuda"):
        from sim.l0star import L0STAR_RHO
        from sim.windows import split_windows
        self.wins = split_windows("train")
        assert (self.wins["split"].astype(str) == "train").all()
        self.segs_per_window, self.per_seg, self.device, self.key = segs_per_window, per_seg, device, seed_key
        self.rho_values = tuple(float(r) for r in L0STAR_RHO)
        self.k = 0
        self.visits = 0
        self.skipped_small = 0

    def segment_loci(self):
        """One window visit -> list of (segment, locus)."""
        from sim.generator import WindowContext
        from sim.l0star import make_segment, sample_locus
        out = []
        while not out:
            g = rng(REPLICATE_SEED, "famC", "c1", self.key, self.k)
            self.k += 1
            w = self.wins.iloc[int(g.integers(len(self.wins)))]
            ctx = WindowContext(w["chr"], int(w["window_start"]), int(w["window_end"]), split="train",
                                block_ids=tuple(w["block_ids"]), anchor_block=str(w["block_id"]),
                                anchor=int(w["anchor"]), device=self.device)
            self.visits += 1
            if ctx.p < MIN_WINDOW_P:
                self.skipped_small += 1
                ctx.release()
                continue
            for _ in range(self.segs_per_window):
                seg = make_segment(ctx, g)
                if seg is None:
                    continue
                out += [(seg, sample_locus(seg, g, self.rho_values[int(g.integers(len(self.rho_values)))], KINDS))
                        for _ in range(self.per_seg)]
            ctx.release()
        return out


def train(a):
    out = OUT / a.run
    out.mkdir(parents=True, exist_ok=True)
    from sim import l0star
    cfg = dict(vars(a), logit_correction="none", p=l0star.P_DEFAULT, eta=l0star.ETA, rho_values=list(l0star.L0STAR_RHO),
               n_nominal=list(N_NOMINAL), kinds=list(KINDS), min_window_p=MIN_WINDOW_P, split="train",
               type_prior=l0star.TYPE_PRIOR,
               lambda_mix={t: l0star.LAMBDA_MIX[t][0].tolist() for t in (1, 2)})
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    torch.manual_seed(1)
    model = NeuralColoc(d=a.d, layers=a.layers, K=a.K).cuda()
    print(f"params {n_params(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.n_train // a.batch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) *
                                              (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    stream = L0StarStream(a.run, per_seg=a.per_seg, segs_per_window=a.segs_per_window)
    pool: list = []
    seen, t0 = 0, time.time()
    gaug = torch.Generator(device="cuda")
    gaug.manual_seed(int(rng(REPLICATE_SEED, "famC", "c1aug", a.run).integers(2 ** 62)))
    gperm = rng(REPLICATE_SEED, "famC", "c1perm", a.run)
    evals = sorted(int(x) for x in a.eval_at.split(",")) if a.eval_at else []
    log = []
    stopped = None
    for step in range(1, steps + 1):
        while len(pool) < a.batch * a.pool_batches:
            pool += stream.segment_loci()
        perm = gperm.permutation(len(pool))
        take = [pool[i] for i in perm[: a.batch]]
        pool = [pool[i] for i in perm[a.batch:]]
        segs = [x[0] for x in take]
        loci = [x[1] for x in take]
        fl = torch.where(torch.rand(a.batch, 2, generator=gaug, device="cuda") < 0.5, -1.0, 1.0).double()
        y = torch.tensor([L0S_CLASSES.index(x["oracle"].cls) for x in loci], device="cuda")
        for attempt in range(20):        # shared GPU: wait and retry the same batch on OOM (never skip it)
            try:
                b, ld = l0star_batch(segs, loci)
                f = build(extract(b, ld, M=a.M), fl)
                o = model(f)
                loss = torch.nn.functional.cross_entropy(o["logits"], y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                break
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"OOM at step {step}, retry {attempt + 1}", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError("repeated CUDA OOM")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        seen += a.batch
        el = time.time() - t0
        if step % 200 == 0:
            print(f"step {step} seen {seen} ce {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{seen / el:.0f} loci/s visits {stream.visits} el {el / 60:.1f}m", flush=True)
            log.append(dict(step=step, seen=seen, ce=float(loss.item()), elapsed_s=el))
        if evals and seen >= evals[0]:
            evals.pop(0)
            torch.save(model.state_dict(), out / f"model_{seen}.pt")
            print(f"checkpoint model_{seen}.pt at {el / 60:.1f} min", flush=True)
        if a.max_hours and el > a.max_hours * 3600:
            stopped = dict(reason=f"time budget {a.max_hours} h reached", seen=seen, step=step)
            print(f"STOP: {stopped}", flush=True)
            break
    torch.save(model.state_dict(), out / "model.pt")
    (out / "train_log.json").write_text(json.dumps(dict(seen=seen, elapsed_s=time.time() - t0, stopped=stopped,
                                                        window_visits=stream.visits,
                                                        skipped_small_windows=stream.skipped_small, log=log), indent=1))
    print(f"done: seen {seen} in {(time.time() - t0) / 60:.1f} min; final model.pt", flush=True)


def checkpoint_config(checkpoint) -> dict:
    p = Path(checkpoint).parent / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def predict_l0star(checkpoint, set_path, class_w="auto", d=128, layers=4, K=16, batch=64) -> "pd.DataFrame":
    """5-class natural-mixture posteriors of a checkpoint on an L0* locus set (no truth access).
    class_w = "auto": the main-task correction only if the run's config.json records main-task class weights
    (`class_w`, famC.train_main); no correction for famC.c1 runs (every famC.c1 trainer, c1_pilot and c1_long
    included, used unweighted CE on the natural stream)."""
    import pandas as pd
    from eval.io import iter_loci
    from famC.method import NeuralColoc as Meth
    if class_w == "auto":
        cfg = checkpoint_config(checkpoint)
        class_w = "main" if ("class_w" in cfg and cfg.get("logit_correction") != "none") else None
    loci = list(iter_loci(set_path))
    rows = Meth(checkpoint, d=d, layers=layers, K=K, batch=batch, class_w=class_w).fit_batch(loci)
    bad = [r["locus_id"] for r in rows if r["status"] != "ok"]
    if bad:
        raise RuntimeError(f"{len(bad)} loci failed, e.g. {bad[:3]}")
    return pd.DataFrame({"locus_id": [r["locus_id"] for r in rows],
                         **{k: [json.loads(r["extra"])["P5"][k] for r in rows] for k in L0S_CLASSES}})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--run", required=True)
    t.add_argument("--n-train", type=int, default=16_000_000)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--d", type=int, default=128)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--K", type=int, default=16)
    t.add_argument("--M", type=int, default=128)
    t.add_argument("--eval-at", default="")
    t.add_argument("--per-seg", type=int, default=128)
    t.add_argument("--segs-per-window", type=int, default=32)
    t.add_argument("--pool-batches", type=int, default=64)
    t.add_argument("--max-hours", type=float, default=0.0, help="stop and save model.pt after this wall time")
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()

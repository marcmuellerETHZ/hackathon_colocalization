"""C0 kill-test training and evaluation (DESIGN v4 §5 C).

    python -m famC.train_c0 --run c0_full --steps 40000            # full model
    python -m famC.train_c0 --run c0_screen --variant screen        # ablation: screen stream only
    python -m famC.train_c0 --run c0_nold --variant nold            # ablation: no LD

Training: on-the-fly loci from TRAIN-split windows (LDBank "train"), hypotheses uniform over H0..H4 (README C0-2),
targets = coloc.abf PP (W = 0.15, p1 = p2 = 1e-4, p12 = 1e-5) over the full window. Random draws come from one
CUDA generator seeded with derive(20261030, "famC", "C0train", run) (DESIGN §9 replicate/training seed).
Evaluation: fixed VALIDATION-split windows (LDBank "valid"), seed 20261029 (DESIGN §9 NPE validation draws), two
hypothesis distributions: uniform and coloc's own prior. Criterion (§5 C): E[KL(coloc || net)] < 0.01 and 99th
percentile |dPP4| < 0.05 on validation loci. The same metrics on train-split windows (fresh draws) give the gap.
Outputs: results/famC/<run>/{curve.tsv, metrics.json, model.pt, config.json}; log to stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from common.seeds import REPLICATE_SEED, VALID_SEED
from famC.c0world import sample_c0
from famC.features import FactorLD, featurize
from famC.ldbank import LDBank, torch_gen
from famC.model import NeuralColoc, n_params

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "famC"


def kl_rows(t: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    t = t.double()
    logq = logq.double()
    lt = torch.where(t > 0, torch.log(t.clamp_min(1e-300)), torch.zeros_like(t))
    return (t * (lt - logq)).sum(-1)


def pip_loss(out, f, b):
    sel, valid = f["sel"], f["valid"]
    t_tok = torch.gather(b["snp_pp4"], 1, sel) * valid
    t_null = (1 - t_tok.sum(-1, keepdim=True)).clamp_min(0)
    t = torch.cat([t_tok, t_null], -1).float()
    lp = torch.log_softmax(out["pip_logits"].masked_fill(torch.cat([~valid, torch.zeros_like(valid[:, :1])], -1),
                                                         -1e9), -1)
    ce = -(t * lp).sum(-1)
    w = b["pp"][:, 4].float()
    return (w * ce).mean()


def batch_and_features(bank, g, gc, wb, L, hyp, variant, aug=True, widx=None):
    b = sample_c0(bank, g, gc, wb, L, hyp_probs=hyp, widx=widx)
    B = b["z1"].shape[0]
    fl = None
    if aug:
        fl = torch.where(torch.rand(B, 2, generator=g, device=g.device) < 0.5, -1.0, 1.0).double()
    f = featurize(b, FactorLD(b["X"], L), M=128, use_ld=(variant != "nold"), flip_traits=fl)
    return b, f


@torch.no_grad()
def evaluate(model, bank, variant, dist, n_sweeps=2, L=25, wb=10, seed_key="val"):
    model.eval()
    g = torch_gen(VALID_SEED, "C0eval", seed_key, dist, device="cuda")
    gc = torch_gen(VALID_SEED, "C0evalc", seed_key, dist)
    kls, d4, dall, agree, ent = [], [], [], [], []
    n = 0
    t0 = time.time()
    t_feat = 0.0
    for _ in range(n_sweeps):
        for widx in bank.all_batches(wb):
            b = sample_c0(bank, g, gc, len(widx), L, hyp_probs=dist, widx=widx)
            torch.cuda.synchronize()
            t1 = time.time()
            f = featurize(b, FactorLD(b["X"], L), M=128, use_ld=(variant != "nold"))
            out = model(f)
            torch.cuda.synchronize()
            t_feat += time.time() - t1
            logq = torch.log_softmax(out["logits"].double(), -1)
            q = logq.exp()
            kls.append(kl_rows(b["pp"], logq).cpu())
            d4.append((q[:, 4] - b["pp"][:, 4]).abs().cpu())
            dall.append((q - b["pp"]).abs().cpu())
            agree.append((q.argmax(-1) == b["pp"].argmax(-1)).cpu())
            n += q.shape[0]
    kl = torch.cat(kls).numpy()
    d4 = torch.cat(d4).numpy()
    dall = torch.cat(dall).numpy()
    model.train()
    res = dict(n=n, E_KL=float(kl.mean()), KL_q50=float(np.quantile(kl, .5)), KL_q90=float(np.quantile(kl, .9)),
               KL_q99=float(np.quantile(kl, .99)), KL_max=float(kl.max()),
               dPP4_q99=float(np.quantile(d4, .99)), dPP4_q999=float(np.quantile(d4, .999)), dPP4_max=float(d4.max()),
               dPP4_mean=float(d4.mean()),
               dPPk_q99=[float(np.quantile(dall[:, k], .99)) for k in range(5)],
               argmax_agree=float(torch.cat(agree).float().mean()),
               infer_loci_per_s=float(n / t_feat))
    res["pass"] = bool(res["E_KL"] < 0.01 and res["dPP4_q99"] < 0.05)
    return res


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", default="full", choices=["full", "screen", "nold"])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--wb", type=int, default=32)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--n-train-windows", type=int, default=3000)
    ap.add_argument("--n-val-windows", type=int, default=400)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--pip-weight", type=float, default=0.1)
    ap.add_argument("--energy", default="free", choices=["free", "sat"])
    ap.add_argument("--hyp", default="uniform")
    ap.add_argument("--seed-key", default="main")
    ap.add_argument("--eval-at", default="50000,100000,200000,500000,1000000,2000000,5000000,10000000")
    ap.add_argument("--eval-sweeps", type=int, default=2)
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop after the evaluation at this many training loci (same lr schedule as --steps)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out_dir = OUT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args) | {"git": git_hash(), "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    t0 = time.time()
    tr = LDBank("train", args.n_train_windows, REPLICATE_SEED, verbose=True)
    va = LDBank("valid", args.n_val_windows, VALID_SEED, verbose=True)
    print(f"banks: train {len(tr)} windows ({tr.nbytes() / 1e9:.2f} GB), valid {len(va)}; {time.time() - t0:.0f}s",
          flush=True)
    model = NeuralColoc(d=args.d, layers=args.layers, K=args.K, use_ld=args.variant != "nold",
                        use_all=args.variant != "screen", energy=args.energy).cuda()
    print(f"params {n_params(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    g = torch_gen(REPLICATE_SEED, "C0train", args.run, args.seed_key, device="cuda")
    gc = torch_gen(REPLICATE_SEED, "C0trainc", args.run, args.seed_key)
    evals = sorted(int(x) for x in args.eval_at.split(","))
    curve = []
    seen = 0
    tl, ttime = [], time.time()
    hyp = args.hyp if args.hyp in ("uniform", "coloc") else json.loads(args.hyp)
    if args.stop_after:
        evals = [e for e in evals if e <= args.stop_after]
    for step in range(1, args.steps + 1):
        b, f = batch_and_features(tr, g, gc, args.wb, args.L, hyp, args.variant)
        out = model(f)
        logq = torch.log_softmax(out["logits"].double(), -1)
        loss = kl_rows(b["pp"], logq).mean()
        lp = pip_loss(out, f, b)
        tot = loss.float() + args.pip_weight * lp
        opt.zero_grad(set_to_none=True)
        tot.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        seen += b["z1"].shape[0]
        tl.append(loss.item())
        if step % 200 == 0:
            el = time.time() - ttime
            print(f"step {step} seen {seen} KL {np.mean(tl):.5f} pip {lp.item():.4f} gn {gn:.2f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {200 * b['z1'].shape[0] / el:.0f} loci/s "
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
            tl, ttime = [], time.time()
        last = step == args.steps or bool(args.stop_after and seen >= args.stop_after)
        while evals and seen >= evals[0] or last:
            if evals and seen >= evals[0]:
                evals.pop(0)
            rec = {"step": step, "seen": seen, "time_s": time.time() - t0}
            for dist in ("uniform", "coloc"):
                rec[f"val_{dist}"] = evaluate(model, va, args.variant, dist, n_sweeps=args.eval_sweeps)
            rec["train_uniform"] = evaluate(model, tr, args.variant, "uniform", n_sweeps=1, L=4, wb=25,
                                            seed_key="trainregion")
            curve.append(rec)
            vu, vc = rec["val_uniform"], rec["val_coloc"]
            print(f"EVAL seen {seen}: uniform E_KL {vu['E_KL']:.5f} dPP4_q99 {vu['dPP4_q99']:.4f} | coloc E_KL "
                  f"{vc['E_KL']:.5f} dPP4_q99 {vc['dPP4_q99']:.4f} | train E_KL {rec['train_uniform']['E_KL']:.5f}",
                  flush=True)
            (out_dir / "curve.json").write_text(json.dumps(curve, indent=1))
            torch.save(model.state_dict(), out_dir / "model.pt")
            if last:
                break
        if last:
            break
    with open(out_dir / "curve.tsv", "w") as fh:
        fh.write("seen\tstep\tval_uniform_E_KL\tval_uniform_dPP4_q99\tval_coloc_E_KL\tval_coloc_dPP4_q99\t"
                 "train_uniform_E_KL\ttrain_uniform_dPP4_q99\n")
        for r in curve:
            fh.write(f"{r['seen']}\t{r['step']}\t{r['val_uniform']['E_KL']:.6g}\t{r['val_uniform']['dPP4_q99']:.6g}\t"
                     f"{r['val_coloc']['E_KL']:.6g}\t{r['val_coloc']['dPP4_q99']:.6g}\t"
                     f"{r['train_uniform']['E_KL']:.6g}\t{r['train_uniform']['dPP4_q99']:.6g}\n")
    final = curve[-1]
    metrics = {"run": args.run, "variant": args.variant, "final": final, "params": n_params(model),
               "train_loci": seen, "wall_s": time.time() - t0,
               "C0_pass": bool(final["val_uniform"]["pass"] and final["val_coloc"]["pass"])}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("DONE", json.dumps({k: metrics[k] for k in ("C0_pass", "train_loci", "wall_s")}), flush=True)


if __name__ == "__main__":
    main()

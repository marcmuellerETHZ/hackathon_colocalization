"""Family C main-task training on simulator records (5 oracle classes S, D, W, G, U + shared-PIP head).

    python -m famC.train_main --run main_v1 --train-set train_v1 --valid-set valid_v1 --steps 30000
    python -m famC.train_main --run main_nold --variant nold ...      # no-LD ablation

Data: shards from famC.gen_shards (train split, L6 held out; validation split with all levels incl. L6).
Loss: class-weighted cross-entropy (weights CLASS_W, README C-T2) + 0.1 x shared-PIP cross-entropy (token of the
class-determining shared variant, or 'none'); trait-sign augmentation (both traits independently, rho flipped
accordingly). Inference: logits - log CLASS_W (natural-mixture posteriors, `natural_logits`).
Monitoring metrics (not the pre-registered evaluation, which goes through src/eval): S-vs-D AUPRC/AUROC of
s = P_S/(P_S + P_D) (power-matched per level with eval.metrics.power_match_weights, and unweighted), D-vs-W AUROC
of P_D, 5-class log loss and accuracy; per level, with L6 = held-out mechanism.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch

from common.seeds import REPLICATE_SEED, derive
from eval.metrics import average_precision, power_match_weights, roc_auc
from famC.features import build
from famC.ldbank import torch_gen
from famC.model import NeuralColoc, n_params
from famC.simadapter import CLASS_ORDER, DATA_DIR, META_COLS, collate_records

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "famC"
CLASS_W = torch.tensor([1.0, 1.0, 0.12, 0.4, 1.0])       # S, D, W, G, U  (~ inverse natural frequency, capped)
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L6")             # sim.train_api.LEVELS order (meta 'level' indexes it)
MI = {c: i for i, c in enumerate(META_COLS)}


def list_shards(name: str) -> list[Path]:
    s = sorted((DATA_DIR / name).glob("shard*.pt"))
    if not s:
        raise FileNotFoundError(f"no shards in {DATA_DIR / name}")
    return s


def load_all(name: str, max_loci: int | None = None):
    recs, metas = [], []
    for f in list_shards(name):
        d = torch.load(f, weights_only=False)
        recs += d["recs"]
        metas.append(d["meta"])
        if max_loci and len(recs) >= max_loci:
            break
    meta = torch.cat(metas)
    if max_loci:
        recs, meta = recs[:max_loci], meta[:max_loci]
    return recs, meta


class ShardStream:
    """Background loader: shards in a seeded random order (re-shuffled every epoch, new shards picked up),
    shuffle buffer of `buf_shards` shards, yields (records, meta) batches on CPU."""

    def __init__(self, name: str, batch: int, seed_key: str, buf_shards: int = 6, max_shards: int | None = None):
        self.name, self.batch, self.buf_shards, self.max_shards = name, batch, buf_shards, max_shards
        self.rng = random.Random(derive(REPLICATE_SEED, "famC", "stream", seed_key))
        self.q: queue.Queue = queue.Queue(maxsize=8)
        self.epoch = 0
        self.loci_read = 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        buf_r, buf_m = [], None
        while True:
            shards = list_shards(self.name)
            if self.max_shards:
                shards = shards[: self.max_shards]
            self.rng.shuffle(shards)
            for f in shards:
                d = torch.load(f, weights_only=False)
                buf_r += d["recs"]
                buf_m = d["meta"] if buf_m is None else torch.cat([buf_m, d["meta"]])
                self.loci_read += len(d["recs"])
                if len(buf_r) < self.buf_shards * len(d["recs"]):
                    continue
                perm = list(range(len(buf_r)))
                self.rng.shuffle(perm)
                n_out = len(buf_r) // 2
                take, keep = perm[:n_out], perm[n_out:]
                for i in range(0, n_out - self.batch + 1, self.batch):
                    idx = take[i:i + self.batch]
                    self.q.put(([buf_r[j] for j in idx], buf_m[idx]))
                buf_r = [buf_r[j] for j in keep]
                buf_m = buf_m[keep]
            self.epoch += 1

    def __iter__(self):
        while True:
            yield self.q.get()


def natural_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - torch.log(CLASS_W.to(logits.device))


def strip_ld(f: dict) -> dict:
    """No-LD ablation inputs: zero every LD-derived input (conditional z, LD with the leads, eigen-spectrum,
    lead-lead r, all-variant r²) and drop the LD matrix. The screen membership itself was built with the
    conditional lists (README C-A4)."""
    f = dict(f)
    tok = f["tok"].clone()
    tok[..., 4:10] = 0
    f["tok"] = tok
    q = f["q_tok"].clone(); q[..., 3:] = 0; f["q_tok"] = q
    q = f["q_all"].clone(); q[..., 3:] = 0; f["q_all"] = q
    a = f["all_feat"].clone(); a[..., 7:9] = 0; f["all_feat"] = a
    g = f["glob"].clone(); g[:, 14:17] = 0; g[:, 20:] = 0; f["glob"] = g
    f["Rmm"] = None
    return f


def forward_batch(model, recs, meta, variant, gen=None, aug=True, device="cuda"):
    rec = collate_records(recs, device)
    B = len(recs)
    fl = None
    if aug:
        fl = torch.where(torch.rand(B, 2, generator=gen, device=gen.device) < 0.5, -1.0, 1.0).double().to(device)
    f = build(rec, fl)
    if variant == "nold":
        f = strip_ld(f)
    return model(f)


def losses(out, meta, pip_weight=0.1):
    dev = out["logits"].device
    y = meta[:, MI["y"]].long().to(dev)
    ce = torch.nn.functional.cross_entropy(out["logits"], y, weight=CLASS_W.to(dev))
    pt = meta[:, MI["pip_t"]].long().to(dev)
    pl = torch.nn.functional.cross_entropy(out["pip_logits"], pt)
    return ce, pl, ce + pip_weight * pl


@torch.no_grad()
def predict(model, recs, meta, variant, bs=256):
    model.eval()
    probs, t_feat = [], 0.0
    for i in range(0, len(recs), bs):
        torch.cuda.synchronize()
        t0 = time.time()
        out = forward_batch(model, recs[i:i + bs], meta[i:i + bs], variant, aug=False)
        p = torch.softmax(natural_logits(out["logits"].double()), -1)
        torch.cuda.synchronize()
        t_feat += time.time() - t0
        probs.append(p.cpu())
    model.train()
    return torch.cat(probs).numpy(), len(recs) / t_feat


def metrics_from_probs(P: np.ndarray, meta: torch.Tensor, w_class=None) -> dict:
    m = meta.numpy()
    y = m[:, MI["y"]].astype(int)
    cls = np.array(CLASS_ORDER)[y]
    level = m[:, MI["level"]].astype(int)
    PS, PD = P[:, 0], P[:, 1]
    s = (PS + 1e-12) / (PS + PD + 2e-12)
    out = {}
    for name, sel in [("all", np.ones(len(y), bool)), ("realistic_L1-L4", (level >= 1) & (level <= 4))] + \
            [(LEVELS[k], level == k) for k in range(len(LEVELS))]:
        if sel.sum() == 0:
            continue
        r = {"n": int(sel.sum())}
        sd = sel & ((cls == "S") | (cls == "D"))
        if (cls[sd] == "S").any() and (cls[sd] == "D").any():
            ysd = cls[sd] == "S"
            r["auprc_sd_unw"] = average_precision(ysd, s[sd])
            r["auroc_sd_unw"] = roc_auc(ysd, s[sd])
            w = power_match_weights(cls[sd], m[sd, MI["mu1"]], m[sd, MI["mu2"]], np.zeros(sd.sum()), level[sd])
            r["auprc_sd_pm"] = average_precision(ysd, s[sd], w)
        dw = sel & ((cls == "D") | (cls == "W"))
        if (cls[dw] == "D").any() and (cls[dw] == "W").any():
            r["auroc_dw"] = roc_auc(cls[dw] == "D", PD[dw])
        r["logloss_nat"] = float(-np.mean(np.log(np.clip(P[sel, y[sel]], 1e-12, 1))))
        r["acc"] = float((P[sel].argmax(1) == y[sel]).mean())
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--train-set", default="train_v1")
    ap.add_argument("--valid-set", default="valid_v1")
    ap.add_argument("--variant", default="full", choices=["full", "nold", "screen"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--energy", default="free", choices=["free", "sat"])
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--n-valid", type=int, default=30000)
    ap.add_argument("--max-train-shards", type=int, default=None)
    ap.add_argument("--seed-key", default="main")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="timing pilot: stop after this many steps and write timing.json (no checkpoint is used)")
    ap.add_argument("--pilot-sizes", default="",
                    help="timing pilot: comma list of size:steps to project (e.g. 75776:2368,600000:18750)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = OUT / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    import os
    shards_used = [f.name for f in list_shards(a.train_set)][: a.max_train_shards]
    (out_dir / "config.json").write_text(json.dumps(vars(a) | {"start": time.strftime("%F %T"),
                                                               "class_w": CLASS_W.tolist(),
                                                               "train_shards": shards_used,
                                                               "commit": os.environ.get("ORCH_COMMIT"),
                                                               "job": os.environ.get("ORCH_JOB_ID")}, indent=1))
    torch.manual_seed(derive(REPLICATE_SEED, "famC", "init", a.run, a.seed_key) % 2 ** 31)
    vrecs, vmeta = load_all(a.valid_set, a.n_valid)
    trecs_eval, tmeta_eval = load_all(a.train_set, 6144)        # train-region monitoring subset
    print(f"valid {len(vrecs)} loci; train-eval {len(trecs_eval)}", flush=True)
    model = NeuralColoc(d=a.d, layers=a.layers, K=a.K, use_ld=a.variant != "nold", use_all=a.variant != "screen",
                        dropout=a.dropout, energy=a.energy, pip_init=0.0).cuda()
    print("params", n_params(model), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) *
                                              (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))
    gen = torch_gen(REPLICATE_SEED, "mainaug", a.run, a.seed_key, device="cuda")
    stream = ShardStream(a.train_set, a.batch, f"{a.run}/{a.seed_key}", max_shards=a.max_train_shards)
    curve, hist = [], []
    t0 = tl = time.time()
    seen = 0
    for step, (recs, meta) in zip(range(1, a.steps + 1), stream):
        for attempt in range(20):        # shared GPU: wait and retry the same batch on OOM (never skip it)
            try:
                out = forward_batch(model, recs, meta, a.variant, gen=gen)
                ce, pl, loss = losses(out, meta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                break
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                out = ce = pl = loss = None
                torch.cuda.empty_cache()
                print(f"OOM at step {step}, retry {attempt + 1}", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError("repeated CUDA OOM")
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        seen += len(recs)
        hist.append((ce.item(), pl.item()))
        if a.stop_after and step == 100:
            t_pilot0, seen_pilot0 = time.time(), seen
        if a.stop_after and step >= a.stop_after:
            rate = (seen - seen_pilot0) / (time.time() - t_pilot0)
            t_eval = time.time()
            Pv, thr = predict(model, vrecs, vmeta, a.variant)
            t_eval = time.time() - t_eval
            proj = {}
            for item in filter(None, a.pilot_sizes.split(",")):
                size, steps = (int(x) for x in item.split(":"))
                train_s = steps * a.batch / rate
                n_eval = math.ceil(steps / max(1, steps // 8))
                proj[size] = {"steps": steps, "train_h": train_s / 3600,
                              "eval_h_at_30k_valid": n_eval * t_eval * (30000 + 6144) / max(len(vrecs), 1) / 3600,
                              "total_h": (train_s + n_eval * t_eval * (30000 + 6144) / max(len(vrecs), 1)) / 3600}
            timing = {"steps_timed": step - 100, "loci_per_s": rate, "batch": a.batch,
                      "eval_s_for_n_valid": t_eval, "n_valid": len(vrecs), "infer_loci_per_s": thr,
                      "max_train_shards": a.max_train_shards, "projection": proj,
                      "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
            (out_dir / "timing.json").write_text(json.dumps(timing, indent=1))
            print("PILOT", json.dumps(timing), flush=True)
            return
        if step % 200 == 0:
            h = np.mean(hist, 0)
            print(f"step {step} seen {seen} ce {h[0]:.4f} pip {h[1]:.4f} gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{200 * a.batch / (time.time() - tl):.0f} loci/s epoch {stream.epoch} read {stream.loci_read}",
                  flush=True)
            hist, tl = [], time.time()
        if step % a.eval_every == 0 or step == a.steps:
            Pv, thr = predict(model, vrecs, vmeta, a.variant)
            Pt, _ = predict(model, trecs_eval, tmeta_eval, a.variant)
            rec = {"step": step, "seen": seen, "epoch": stream.epoch, "time_s": time.time() - t0,
                   "valid": metrics_from_probs(Pv, vmeta), "train_region": metrics_from_probs(Pt, tmeta_eval),
                   "infer_loci_per_s": thr}
            curve.append(rec)
            v = rec["valid"]
            msg = " | ".join(f"{k}: sd_pm {v[k].get('auprc_sd_pm', float('nan')):.4f} dw {v[k].get('auroc_dw', float('nan')):.4f}"
                             for k in ("all", "realistic_L1-L4", "L6") if k in v)
            print(f"EVAL step {step}: {msg} | train-region sd_pm {rec['train_region']['all'].get('auprc_sd_pm', float('nan')):.4f}"
                  f" | ll {v['all']['logloss_nat']:.4f} | infer {thr:.0f} loci/s", flush=True)
            (out_dir / "curve.json").write_text(json.dumps(curve, indent=1))
            torch.save(model.state_dict(), out_dir / "model.pt")
            np.save(out_dir / "valid_probs.npy", Pv)
    (out_dir / "metrics.json").write_text(json.dumps({"final": curve[-1], "params": n_params(model),
                                                      "train_loci_seen": seen, "wall_s": time.time() - t0}, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Re-evaluate a C0 checkpoint on the fixed validation draws (same seeds as train_c0.evaluate) and break the
errors down by hypothesis, lead |z| and sample size; also times inference.

    python -m famC.c0_analyze --run c0_full [--variant full] [--sweeps 2]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from common.seeds import VALID_SEED
from famC.c0world import sample_c0
from famC.features import FactorLD, featurize
from famC.ldbank import LDBank, torch_gen
from famC.model import NeuralColoc
from famC.train_c0 import OUT, kl_rows


@torch.no_grad()
def collect(model, bank, variant, dist, n_sweeps=2, L=25, wb=10, seed_key="val"):
    g = torch_gen(VALID_SEED, "C0eval", seed_key, dist, device="cuda")
    gc = torch_gen(VALID_SEED, "C0evalc", seed_key, dist)
    rows = []
    t_inf = 0.0
    n = 0
    for _ in range(n_sweeps):
        for widx in bank.all_batches(wb):
            b = sample_c0(bank, g, gc, len(widx), L, hyp_probs=dist, widx=widx)
            torch.cuda.synchronize()
            t0 = time.time()
            f = featurize(b, FactorLD(b["X"], L), M=128, use_ld=(variant != "nold"))
            out = model(f)
            torch.cuda.synchronize()
            t_inf += time.time() - t0
            logq = torch.log_softmax(out["logits"].double(), -1)
            q = logq.exp()
            mz1 = b["z1"].abs().amax(-1)
            mz2 = b["z2"].abs().amax(-1)
            rows.append(torch.stack([kl_rows(b["pp"], logq), (q[:, 4] - b["pp"][:, 4]).abs(), b["hyp"].double(),
                                     torch.maximum(mz1, mz2), torch.minimum(mz1, mz2), b["n1"][:, 0], b["n2"][:, 0],
                                     b["pp"][:, 4], q[:, 4]], -1).cpu())
            n += q.shape[0]
    return torch.cat(rows).numpy(), n / t_inf


def breakdown(A: np.ndarray) -> dict:
    kl, d4, hyp, zmax, zmin = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    out = {"by_hyp": {}, "by_min_lead_z": {}}
    for h in range(5):
        m = hyp == h
        out["by_hyp"][f"H{h}"] = {"n": int(m.sum()), "E_KL": float(kl[m].mean()), "dPP4_q99": float(np.quantile(d4[m], .99))}
    edges = [0, 3, 5, 8, 15, 30, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (zmin >= lo) & (zmin < hi)
        if m.sum():
            out["by_min_lead_z"][f"[{lo},{hi})"] = {"n": int(m.sum()), "E_KL": float(kl[m].mean()),
                                                    "dPP4_q99": float(np.quantile(d4[m], .99)),
                                                    "frac_dPP4_gt_0.05": float((d4[m] > 0.05).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--energy", default="free")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    va = LDBank("valid", 400, VALID_SEED)
    model = NeuralColoc(d=a.d, layers=a.layers, K=a.K, use_ld=a.variant != "nold", use_all=a.variant != "screen", energy=a.energy).cuda()
    model.load_state_dict(torch.load(OUT / a.run / "model.pt"))
    model.eval()
    res = {}
    for dist in ("uniform", "coloc"):
        A, thr = collect(model, va, a.variant, dist, a.sweeps)
        res[dist] = {"n": int(len(A)), "E_KL": float(A[:, 0].mean()), "dPP4_q99": float(np.quantile(A[:, 1], .99)),
                     "infer_loci_per_s": thr, **breakdown(A)}
        np.save(OUT / a.run / f"analysis_{dist}.npy", A)
    (OUT / a.run / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

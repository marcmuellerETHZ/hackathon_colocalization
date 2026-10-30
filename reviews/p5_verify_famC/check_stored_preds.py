"""p5.verify_famC_scoring, CPU-only check on the STORED famC pilot predictions (results/famC/c1_pilot_eval_v2):
  (a) the 'main' predictions must equal the 'none' predictions re-weighted by 1/CLASS_W (pure logit shift);
  (b) recompute natural / balanced E[KL(B||C)] for every checkpoint and both modes by hand and compare with the
      numbers c1_eval.py stored in c1_eval_pilot_{none,main}.json.
Pilot set only (data/processed/famC/l0star_pilot)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

H = Path("/cluster/scratch/mumarc/hackathon_eqtl_gwas")
EV = H / "results/famC/c1_pilot_eval_v2"
SET = H / "data/processed/famC/l0star_pilot"
CL = ["S", "D", "W", "G", "U"]
CW = np.array([1.0, 1.0, 0.12, 0.4, 1.0])

stats = json.loads((SET / "l0star_pilot.stats.json").read_text())
nat = stats["natural_class_shares"]
truth = pd.read_parquet(SET / "l0star_pilot.truth.parquet").set_index("locus_id")["oracle_class"].astype(str)
bp = pd.read_parquet(SET / "famB_sim_preds.parquet").set_index("locus_id")
ex = bp["extra"].map(json.loads)
B = pd.DataFrame({"S": bp["P_S"], "D": bp["P_D"], "W": ex.map(lambda e: e["P_W_strict"]),
                  "G": ex.map(lambda e: e["P_G"]), "U": ex.map(lambda e: e["P_U"])})
print("B row sums min/max", B.sum(1).min(), B.sum(1).max(), "| B exact zeros per class",
      {c: int((B[c] == 0).sum()) for c in CL})


def kl(PB, PC):
    t = np.zeros_like(PB)
    m = PB > 0
    t[m] = PB[m] * (np.log(PB[m]) - np.log(PC[m]))
    return t.sum(1)


rep = {}
stored = {m: json.loads((EV / f"c1_eval_pilot_{m}.json").read_text())["checkpoints"] for m in ("none", "main")}
for ck in stored["none"]:
    P = {}
    for m in ("none", "main"):
        df = pd.read_parquet(EV / f"c1_eval_pilot_{m}.{ck}.preds.parquet").set_index("locus_id")
        P[m] = df.loc[B.index, CL].to_numpy(float)
    # (a) main == normalise(none / CW)
    q = P["none"] / CW
    q /= q.sum(1, keepdims=True)
    max_dev = float(np.abs(q - P["main"]).max())
    cls = truth.loc[B.index].to_numpy()
    PB = B.to_numpy(float)
    out = {"max_abs(main - renorm(none/CW))": max_dev}
    for m in ("none", "main"):
        PC = np.clip(P[m], 1e-12, 1)   # stored P5 is rounded to 8 decimals; clip only guards exact zeros
        k = kl(PB, PC)
        present = ["S", "D", "W", "G"]
        z = sum(nat[c] for c in present)
        natural = sum(nat[c] / z * k[cls == c].mean() for c in present)
        s = stored[m][ck]
        out[m] = {"natural_mine": natural, "natural_c1_eval": s["C1_E_KL_natural"],
                  "balanced_mine": float(k.mean()), "balanced_c1_eval": s["E_KL_balanced"],
                  "min_PC": float(P[m].min())}
    rep[ck] = out
print(json.dumps(rep, indent=1))
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(rep, indent=1))

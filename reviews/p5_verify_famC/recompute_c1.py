"""p5.verify_famC_scoring: independent recomputation of the C1 statistic on the 600-locus PILOT set
(data/processed/famC/l0star_pilot; NOT l0star_v2) for the c1_long 8 M checkpoint.

Independent of famC.method / scripts/famC_eval/c1_eval.py:
  * inputs: each LocusData is turned into a famC batch dict by hand, in the same form as the C1 *training* stream
    (famC.c1.l0star_batch: type1 = 0, type2 = 1, rho_hat from the set, dense R~), one locus at a time (no padding,
    no StackedLD), then famC.features.extract/build and the network -> RAW logits (float64);
  * posterior: plain softmax of the raw logits (the correct transform for an unweighted-CE model on the natural
    stream), and, for the record, softmax(logits - log CLASS_W) with CLASS_W = (1, 1, 0.12, 0.4, 1) typed in here
    (the M3.5 "main" correction);
  * B: Family B simulator-mode rows parsed here from famB_sim_preds.parquet (P_S, P_D, extra.P_W_strict/P_G/P_U);
  * KL(B || C) = sum_k B_k (ln B_k - ln C_k), natural log, 0 ln 0 = 0, no clipping of C (C > 0 checked);
  * natural expectation written as sum_c nat'_c * mean_{loci of class c} KL with nat' the stats' natural shares
    renormalised over the classes present (S, D, W, G), which equals c1_eval's per-locus-weight form.
Outputs a JSON with per-locus values (50-locus stratified subset in detail, plus full-set statistics).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CL = ("S", "D", "W", "G", "U")
CLASS_W_OLD = np.array([1.0, 1.0, 0.12, 0.4, 1.0])


def raw_logits(ckpt, loci, device="cuda"):
    from famC.features import DenseLD, build, extract
    from famC.model import NeuralColoc
    net = NeuralColoc(d=128, layers=4, K=16).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    out = []
    with torch.no_grad():
        for lc in loci:
            t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)[None]
            p = len(lc.z1)
            b = dict(z1=t(lc.z1), z2=t(lc.z2), se1=t(lc.se1), se2=t(lc.se2), n1=t(lc.n1), n2=t(lc.n2),
                     maf1=t(lc.maf1), maf2=t(lc.maf2), maf_ref=t(lc.maf_ref),
                     mask=torch.ones(1, p, dtype=torch.bool, device=device),
                     n_ref=torch.tensor([float(lc.n_ref)], dtype=torch.float64, device=device),
                     rho_hat=torch.tensor([float(lc.rho_hat)], dtype=torch.float64, device=device),
                     rho_se=torch.tensor([float(lc.rho_se)], dtype=torch.float64, device=device),
                     type1=torch.zeros(1, device=device), type2=torch.ones(1, device=device))
            R = torch.as_tensor(np.asarray(lc.R, dtype=np.float32), device=device)[None]
            o = net(build(extract(b, DenseLD(R), M=128)))
            out.append(o["logits"].double().cpu().numpy()[0])
    return np.array(out)


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def kl(PB, PC):
    assert (PC > 0).all()
    t = np.zeros_like(PB)
    m = PB > 0
    t[m] = PB[m] * (np.log(PB[m]) - np.log(PC[m]))
    return t.sum(1)


def stat(klv, cls, nat):
    present = [c for c in CL if (cls == c).any()]
    z = sum(nat[c] for c in present)
    return {"natural": float(sum(nat[c] / z * klv[cls == c].mean() for c in present)),
            "balanced": float(klv.mean()),
            "by_class": {c: float(klv[cls == c].mean()) for c in present}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="data/processed/famC/l0star_pilot/l0star_pilot.h5")
    ap.add_argument("--b", default="data/processed/famC/l0star_pilot/famB_sim_preds.parquet")
    ap.add_argument("--ckpt", default="results/famC/c1_long/model.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-detail", type=int, default=50)
    a = ap.parse_args()
    from eval.io import iter_loci
    sp = Path(a.set)
    assert "l0star_v2" not in str(sp)
    stats = json.loads(Path(str(sp).replace(".h5", ".stats.json")).read_text())
    truth = pd.read_parquet(str(sp).replace(".h5", ".truth.parquet"))[["locus_id", "oracle_class"]]
    nat = stats["natural_class_shares"]
    # natural shares recomputed from the raw candidate counts (check of the stats field)
    cc = stats["candidate_class_counts"]
    nat_cc = {c: v / sum(cc.values()) for c, v in cc.items()}
    loci = list(iter_loci(sp))
    ids = [lc.locus_id for lc in loci]
    L = raw_logits(a.ckpt, loci)
    bp = pd.read_parquet(a.b).set_index("locus_id").loc[ids]
    assert (bp["status"] == "ok").all()
    ex = bp["extra"].map(json.loads)
    PB = np.stack([bp["P_S"].to_numpy(float), bp["P_D"].to_numpy(float),
                   ex.map(lambda e: e["P_W_strict"]).to_numpy(float), ex.map(lambda e: e["P_G"]).to_numpy(float),
                   ex.map(lambda e: e["P_U"]).to_numpy(float)], 1)
    cls = truth.set_index("locus_id").loc[ids, "oracle_class"].astype(str).to_numpy()
    PC_none = softmax(L)
    PC_old = softmax(L - np.log(CLASS_W_OLD))
    k_none, k_old = kl(PB, PC_none), kl(PB, PC_old)
    # 50-locus stratified detail subset (seeded)
    g = np.random.default_rng(20261027)
    det = []
    for c, n in zip(("S", "D", "W", "G"), (13, 13, 12, 12)):
        det += list(g.choice(np.where(cls == c)[0], n, replace=False))
    det = sorted(det)[: a.n_detail]
    res = {"set": str(sp), "ckpt": a.ckpt, "b": a.b, "n": len(ids),
           "B_row_sum_range": [float(PB.sum(1).min()), float(PB.sum(1).max())],
           "natural_shares_stats": nat, "natural_shares_from_counts": nat_cc,
           "min_PC_none": float(PC_none.min()), "min_PC_old": float(PC_old.min()),
           "full_none": stat(k_none, cls, nat), "full_old": stat(k_old, cls, nat),
           "subset50_none": stat(k_none[det], cls[det], nat), "subset50_old": stat(k_old[det], cls[det], nat),
           "loci": [{"locus_id": ids[i], "cls": cls[i], "logits": L[i].tolist(), "PB": PB[i].tolist(),
                     "PC_none": PC_none[i].tolist(), "PC_old": PC_old[i].tolist(),
                     "KL_none": float(k_none[i]), "KL_old": float(k_old[i]), "detail": bool(i in set(det))}
                    for i in range(len(ids))]}
    # sensitivity of the natural statistic to the MC error of the natural shares (+-2 SE on each class)
    se = stats["natural_class_shares_mc_se"]
    sens = {}
    for c in ("S", "D", "W", "G"):
        for s in (-2, 2):
            n2 = dict(nat)
            n2[c] = nat[c] + s * se[c]
            sens[f"{c}{s:+d}SE"] = stat(k_none, cls, n2)["natural"]
    res["natural_share_sensitivity_none"] = sens
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "loci"}, indent=1))


if __name__ == "__main__":
    main()

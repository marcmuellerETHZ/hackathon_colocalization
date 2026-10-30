"""C1 evaluation (DESIGN v4 §5 C; DECISIONS P3-4, P3-9, P3-21): KL(B || C) over the 5 oracle classes on an L0*
validation locus set, against Family B's simulator-prior posteriors (the Bayes-optimal posterior in L0*).

This is EVALUATION code: it reads the set's truth (oracle labels) and stats (natural class shares), so it lives
outside the method package src/famC (P3-21). The network predictions come from `famC.c1.predict_l0star`, which
never reads truth.

C1 statistic (P3-4): E[KL(B || C)] under the world's natural class distribution. The L0* sets are class-stratified
(S/D/W/G x 500, U too rare to fill), so each locus of class c gets weight natural_share[c] / set_share[c]; U is not
in the set, so the expectation is over S, D, W, G with their natural shares renormalised (U's natural share is
reported). PASS iff the natural-weighted E[KL] < 0.02. Secondary: class-balanced (unweighted) E[KL], per-class
E[KL], natural-weighted and unweighted q90 / q99 of the per-locus KL, max |dP_S|, |dP_D|, a cluster bootstrap CI
(clusters = LD blocks of the truth table), and calibration against the oracle labels for C and for B (natural-
weighted log loss and accuracy, one-vs-rest ECE per class with 10 equal-width bins).

    python scripts/famC_eval/c1_eval.py --set data/processed/sets/l0star_v2.h5 \
        --b-preds results/famB/l0star_ref_v1/posteriors.parquet \
        --ckpt results/famC/c1_v2/model_1000448.pt ... --out results/famC/c1_v2/c1_eval_l0star_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CLASSES = ("S", "D", "W", "G", "U")
BAR = 0.02


def load_b(path) -> pd.DataFrame:
    """B posteriors as columns S, D, W, G, U. Formats: famB.l0star_ref (P_S, P_U, P_D, P_W (strict), P_G) or the
    INTERFACES §3 rows of famB.method (P_S, P_D, extra JSON with P_U, P_G, P_W_strict)."""
    bp = pd.read_parquet(path)
    if "status" in bp:
        bad = bp["status"] != "ok"
        if bad.any():
            raise RuntimeError(f"B has {int(bad.sum())} non-ok loci")
    if "P_G" in bp.columns:
        out = pd.DataFrame({"locus_id": bp["locus_id"], "S": bp["P_S"], "D": bp["P_D"], "W": bp["P_W"],
                            "G": bp["P_G"], "U": bp["P_U"]})
        se = [c for c in ("se_S", "se_D", "se_W", "se_G", "se_U") if c in bp.columns]
        for c in se:
            out[c] = bp[c].to_numpy(float)
        return out
    ex = bp["extra"].map(json.loads)
    return pd.DataFrame({"locus_id": bp["locus_id"], "S": bp["P_S"], "D": bp["P_D"],
                         "W": ex.map(lambda e: e["P_W_strict"]), "G": ex.map(lambda e: e["P_G"]),
                         "U": ex.map(lambda e: e["P_U"])})


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][min(np.searchsorted(cw, q), len(x) - 1)])


def kl_rows(PB, PC):
    PB = np.clip(PB, 0, 1)
    PB = PB / PB.sum(1, keepdims=True)
    PC = np.clip(PC, 1e-12, 1)
    PC = PC / PC.sum(1, keepdims=True)
    return (np.where(PB > 0, PB * (np.log(np.clip(PB, 1e-300, 1)) - np.log(PC)), 0)).sum(1)


def calibration(P, y, w) -> dict:
    """P [n, 5] (CLASSES order), y class index, w natural weights."""
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(1, keepdims=True)
    ll = -np.log(P[np.arange(len(y)), y])
    acc = (P.argmax(1) == y).astype(float)
    out = {"logloss_natural": float((w * ll).sum() / w.sum()), "logloss_balanced": float(ll.mean()),
           "acc_natural": float((w * acc).sum() / w.sum()), "acc_balanced": float(acc.mean()), "ece_natural": {}}
    bins = np.linspace(0, 1, 11)
    for k, c in enumerate(CLASSES[:4]):
        p = P[:, k]
        t = (y == k).astype(float)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        e = 0.0
        for bb in range(10):
            m = idx == bb
            if m.any():
                wb = w[m].sum()
                e += wb * abs((w[m] * p[m]).sum() / wb - (w[m] * t[m]).sum() / wb)
        out["ece_natural"][c] = float(e / w.sum())
    return out


def evaluate(pred_c: pd.DataFrame, b: pd.DataFrame, truth: pd.DataFrame, nat: dict, n_boot: int, seed: int) -> dict:
    m = b.merge(pred_c, on="locus_id", suffixes=("_B", "_C")).merge(truth, on="locus_id")
    assert len(m) == len(b) == len(pred_c), (len(m), len(b), len(pred_c))
    PB = m[[f"{k}_B" for k in CLASSES]].to_numpy(float)
    PC = m[[f"{k}_C" for k in CLASSES]].to_numpy(float)
    kl = kl_rows(PB, PC)
    cls = m["oracle_class"].astype(str).to_numpy()
    share = pd.Series(cls).value_counts(normalize=True)
    present = [c for c in CLASSES if c in share.index]
    nat_present = sum(nat.get(c, 0.0) for c in present)
    w = np.array([nat.get(c, 0.0) / nat_present / share[c] for c in cls])
    y = np.array([CLASSES.index(c) for c in cls])
    stat = float((w * kl).sum() / w.sum())
    # cluster bootstrap over LD blocks (loci of one block share LD segments)
    blocks = m["block_id"].astype(str).to_numpy()   # truth block_id = anchor LD block of the segment's window
    ub, inv = np.unique(blocks, return_inverse=True)
    g = np.random.default_rng(seed)
    wkl_b = np.bincount(inv, w * kl, len(ub))
    w_b = np.bincount(inv, w, len(ub))
    kl_b = np.bincount(inv, kl, len(ub))
    n_b = np.bincount(inv, None, len(ub))
    boots, boots_bal = [], []
    for _ in range(n_boot):
        s = np.bincount(g.integers(len(ub), size=len(ub)), minlength=len(ub))
        boots.append((s * wkl_b).sum() / (s * w_b).sum())
        boots_bal.append((s * kl_b).sum() / (s * n_b).sum())
    out = {
        "n": int(len(m)), "n_blocks": int(len(ub)),
        "C1_E_KL_natural": stat,
        "C1_E_KL_natural_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
        "pass": bool(stat < BAR), "bar": BAR,
        "E_KL_balanced": float(kl.mean()),
        "E_KL_balanced_ci95": [float(np.quantile(boots_bal, .025)), float(np.quantile(boots_bal, .975))],
        "E_KL_by_class": {c: float(kl[cls == c].mean()) for c in present},
        # decomposition of the C1 statistic: stat = sum_c natural_share[c] * E_KL_by_class[c]
        "C1_natural_terms": {c: float(nat.get(c, 0.0) / nat_present * kl[cls == c].mean()) for c in present},
        "C1_W_term_share": float(nat.get("W", 0.0) / nat_present * kl[cls == "W"].mean() / stat) if "W" in present else None,
        "n_loci_by_class": {c: int((cls == c).sum()) for c in present},
        "n_blocks_by_class": {c: int(len(np.unique(blocks[cls == c]))) for c in present},
        "natural_share_used": {c: nat.get(c, 0.0) / nat_present for c in present},
        "natural_share_U_not_in_set": float(nat.get("U", 0.0)),
        "KL_q90_natural": wquantile(kl, w, .9), "KL_q99_natural": wquantile(kl, w, .99),
        "KL_q90_balanced": float(np.quantile(kl, .9)), "KL_q99_balanced": float(np.quantile(kl, .99)),
        "KL_max": float(kl.max()),
        "max_abs_dPS": float(np.abs(PB[:, 0] - PC[:, 0]).max()),
        "max_abs_dPD": float(np.abs(PB[:, 1] - PC[:, 1]).max()),
        "corr_by_class": {c: float(np.corrcoef(PB[:, k], PC[:, k])[0, 1]) for k, c in enumerate(CLASSES[:4])},
        "calibration_C": calibration(PC, y, w),
        "calibration_B": calibration(PB, y, w),
    }
    se_cols = [f"se_{c}" for c in CLASSES if f"se_{c}" in m.columns]
    if se_cols:
        se = m[se_cols].to_numpy(float)
        P = np.clip(PB[:, [CLASSES.index(c[3:]) for c in se_cols]], 1e-6, 1)
        # second-order expected KL between B's MC estimate and its limit (noise floor of the comparison)
        out["B_mc_kl_floor_natural"] = float((w * (se ** 2 / (2 * P)).sum(1)).sum() / w.sum())
        out["B_mc_se_max"] = float(np.nanmax(se))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--b-preds", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--class-w", default="auto", choices=["auto", "main", "none"],
                    help="logit correction applied to C (auto: from the run config; see famC.c1.predict_l0star)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261029)
    a = ap.parse_args()
    from famC.c1 import predict_l0star
    set_path = Path(a.set)
    stats = json.loads(Path(str(set_path).replace(".h5", ".stats.json")).read_text())
    truth = pd.read_parquet(str(set_path).replace(".h5", ".truth.parquet"))[["locus_id", "oracle_class", "block_id"]]
    nat = stats["natural_class_shares"]
    b = load_b(a.b_preds)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {"set": str(set_path), "b_preds": a.b_preds, "class_w": a.class_w, "natural_class_shares": nat,
           "checkpoints": {}}
    cw = {"auto": "auto", "main": "main", "none": None}[a.class_w]
    for ck in a.ckpt:
        t0 = time.time()
        pc = predict_l0star(ck, set_path, class_w=cw)
        pc.to_parquet(out.with_name(out.stem + f".{Path(ck).stem}.preds.parquet"))
        r = evaluate(pc, b, truth, nat, a.boot, a.seed)
        r["checkpoint"] = ck
        r["predict_s"] = time.time() - t0
        res["checkpoints"][Path(ck).stem] = r
        print(Path(ck).stem, json.dumps({k: r[k] for k in ("C1_E_KL_natural", "C1_E_KL_natural_ci95", "E_KL_balanced",
                                                            "E_KL_by_class", "pass")}), flush=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

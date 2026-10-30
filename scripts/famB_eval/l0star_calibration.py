"""Calibration of Family B's simulator-prior posteriors on an L0* set against the oracle truth labels
(task p4.famB_l0star_ref). Evaluation code: reads truth, so it lives outside src/famB (INTERFACES §2, P3-21).

    PYTHONPATH=src python scripts/famB_eval/l0star_calibration.py \
        --post results/famB/l0star_ref/posteriors.parquet --set data/processed/sets/l0star_v2.h5 \
        --out results/famB/l0star_ref/calibration.json

The L0* set is class-stratified (S, D, W, G x 500; U never stored). B's posteriors are under the natural L0*
prior. Two readings (P3-4):
  natural   loci weighted w_i = pi'(c_i) / (n_c / N), pi' = natural class shares conditioned on c != U
            (stats.json natural_class_shares, renormalised over S, D, W, G); B's posterior is conditioned on
            c != U as well: q_c = P_c / (1 - P_U). This is exactly the natural L0* distribution given c != U.
  balanced  unweighted loci; B's posterior re-weighted to the balanced class prior (0.25 each, U -> 0) by
            famB.priors.reweight_classes (the class-prior correction for rejection sampling).
If B is the exact posterior, E[cross-entropy] = E[entropy of B's posterior] in both readings (Bayes-optimal
log loss), and every reliability bin has observed frequency = mean predicted probability.
Uncertainty: cluster bootstrap over LD blocks (eval.metrics.cluster_bootstrap, 2,000 replicates; weights are
recomputed from the resampled class counts in every replicate).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.seeds import rng
from eval import metrics as M
from eval.io import truth_path_for
from eval.truth import read_truth
from famB.priors import reweight_classes

C4 = ("S", "D", "W", "G")


def _entropy(q):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(q > 0, -q * np.log(q), 0.0)
    return t.sum(1)


def reliability(q, y, w, n_bins=10):
    """Weighted reliability table of a binary event (probability q, outcome y, weights w)."""
    b = np.minimum((q * n_bins).astype(int), n_bins - 1)
    rows, ece = [], 0.0
    W = w.sum()
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        wk = w[m]
        sw = wk.sum()
        pred = float((wk * q[m]).sum() / sw)
        obs = float((wk * y[m]).sum() / sw)
        n_eff = float(sw ** 2 / (wk ** 2).sum())
        sd = np.sqrt(max(pred * (1 - pred), 1e-12) / n_eff)
        rows.append(dict(bin=f"[{k / n_bins:.1f},{(k + 1) / n_bins:.1f})", n=int(m.sum()), n_eff=round(n_eff, 1),
                         weight_share=float(sw / W), mean_pred=pred, obs_freq=obs, z=float((obs - pred) / sd)))
        ece += sw / W * abs(pred - obs)
    return rows, float(ece)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", default="results/famB/l0star_ref/posteriors.parquet")
    ap.add_argument("--set", default="data/processed/sets/l0star_v2.h5")
    ap.add_argument("--out", default="results/famB/l0star_ref/calibration.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args(argv)

    set_path = Path(a.set)
    post = pd.read_parquet(a.post)
    truth = read_truth(truth_path_for(set_path))
    stats = json.loads((set_path.parent / f"{set_path.stem}.stats.json").read_text())
    df = truth.merge(post, on="locus_id", how="outer", indicator=True)
    assert (df["_merge"] == "both").all(), "posteriors and truth must cover the same loci"
    n_mc = int(post["n_mc"].iloc[0])
    eps = 0.5 / n_mc

    cls = df["oracle_class"].to_numpy().astype(str)
    assert set(cls) <= set(C4), set(cls)
    blocks = df["block_id"].to_numpy().astype(str)
    P5 = {c: df[f"P_{c}"].to_numpy() for c in ("S", "U", "D", "W", "G")}
    nat = stats["natural_class_shares"]
    pi_c = np.array([nat[c] for c in C4])
    pi_c = pi_c / pi_c.sum()  # natural shares given c != U
    Y = np.stack([(cls == c).astype(float) for c in C4], 1)
    yi = Y.argmax(1)

    # natural reading: q = P / (1 - P_U)
    q_nat = np.stack([P5[c] for c in C4], 1) / (1 - P5["U"])[:, None]
    # balanced reading: B's posterior re-weighted to the balanced prior over the stored classes
    ratio = {c: 0.25 / nat[c] for c in C4}
    ratio["U"] = 0.0
    rb = reweight_classes(P5, ratio)
    q_bal = np.stack([rb[c] for c in C4], 1)

    def nat_weights(idx):
        cnt = np.bincount(yi[idx], minlength=4) / len(idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            wc = np.where(cnt > 0, pi_c / cnt, 0.0)
        return wc[yi[idx]]

    def stat(idx):
        w = nat_weights(idx)
        lq_n = np.log(np.maximum(q_nat[idx, yi[idx]], eps))
        lq_b = np.log(np.maximum(q_bal[idx, yi[idx]], eps))
        ce_n = -(w * lq_n).sum() / w.sum()
        h_n = (w * _entropy(q_nat[idx])).sum() / w.sum()
        ce_prior_n = -(w * np.log(pi_c[yi[idx]])).sum() / w.sum()
        ce_b = -lq_b.mean()
        h_b = _entropy(q_bal[idx]).mean()
        return np.array([ce_n, h_n, ce_n - h_n, ce_prior_n, ce_b, h_b, ce_b - h_b])

    names = ["ce_natural", "entropy_natural", "ce_minus_entropy_natural", "ce_prior_only_natural",
             "ce_balanced", "entropy_balanced", "ce_minus_entropy_balanced"]
    boot = M.cluster_bootstrap(stat, blocks, n_boot=a.n_boot, rng=rng(20261029, "famB_l0star_calibration"),
                               n_jobs=a.n_jobs)
    summary = {n: {"estimate": float(boot["estimate"][k]), "se": float(boot["se"][k]),
                   "ci95": [float(boot["ci_low"][k]), float(boot["ci_high"][k])]} for k, n in enumerate(names)}

    w_all = nat_weights(np.arange(len(df)))
    rel = {"natural": {}, "balanced": {}}
    ece = {"natural": {}, "balanced": {}}
    mean_check = {"natural": {}, "balanced": {}}
    for k, c in enumerate(C4):
        rel["natural"][c], ece["natural"][c] = reliability(q_nat[:, k], Y[:, k], w_all)
        rel["balanced"][c], ece["balanced"][c] = reliability(q_bal[:, k], Y[:, k], np.ones(len(df)))
        assert abs(ece["balanced"][c] - M.ece(q_bal[:, k], Y[:, k])) < 1e-12  # same definition as eval.metrics
        mean_check["natural"][c] = {"mean_pred": float((w_all * q_nat[:, k]).sum() / w_all.sum()),
                                    "target_share": float(pi_c[k])}
        mean_check["balanced"][c] = {"mean_pred": float(q_bal[:, k].mean()), "target_share": 0.25}
    zs = [r["z"] for rd in rel.values() for c in C4 for r in rd[c] if r["n"] >= 10]
    # per-class log loss (natural reading, unweighted within class)
    per_class = {c: {"n": int((yi == k).sum()),
                     "mean_neg_log_q_true": float(-np.log(np.maximum(q_nat[yi == k, k], eps)).mean()),
                     "mean_q_true": float(q_nat[yi == k, k].mean())} for k, c in enumerate(C4)}
    se_cols = [f"se_{c}" for c in ("S", "U", "D", "W", "G")]
    out = {
        "set": str(set_path), "posteriors": a.post, "n_loci": len(df), "n_mc": n_mc, "eps_floor": eps,
        "n_true_class_below_eps": {"natural": int((q_nat[np.arange(len(df)), yi] < eps).sum()),
                                   "balanced": int((q_bal[np.arange(len(df)), yi] < eps).sum())},
        "natural_class_shares_given_not_U": dict(zip(C4, pi_c.tolist())),
        "natural_U_share": nat.get("U", 0.0), "mean_P_U": float(P5["U"].mean()),
        "log_loss": summary, "n_boot": a.n_boot, "n_blocks": boot["n_clusters"],
        "ece_10bin": ece, "mean_check": mean_check,
        "reliability_max_abs_z_bins_n_ge_10": float(np.max(np.abs(zs))), "reliability_n_bins_checked": len(zs),
        "reliability": rel, "per_class": per_class,
        "mc_error": {"se_max": float(post[se_cols].to_numpy().max()),
                     "se_mean": float(post[se_cols].to_numpy().mean()),
                     "se_p99": float(np.percentile(post[se_cols].to_numpy(), 99))},
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("reliability",)}, indent=1))
    for rd in ("natural", "balanced"):
        print(f"\nreliability ({rd})")
        for c in C4:
            print(c, " | ".join(f"{r['bin']} n={r['n']} pred={r['mean_pred']:.3f} obs={r['obs_freq']:.3f} z={r['z']:+.1f}"
                                for r in rel[rd][c]))


if __name__ == "__main__":
    main()

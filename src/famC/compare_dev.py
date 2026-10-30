"""Paired locus-level bootstrap between Family C development runs on the same validation records
(monitoring only; claims use src/eval). Power-matching weights are recomputed inside every replicate.

    python -m famC.compare_dev --valid-set valid_dev --runs main_snap1_full main_snap1_nold floor_snap1 --n-boot 1000
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from eval.metrics import average_precision, power_match_weights, roc_auc
from famC.simadapter import CLASS_ORDER
from famC.train_main import MI, OUT, load_all


def endpoints(P, cls, mu1, mu2, level, idx):
    c, lv = cls[idx], level[idx]
    PS, PD = P[idx, 0], P[idx, 1]
    s = (PS + 1e-12) / (PS + PD + 2e-12)
    sd = (c == "S") | (c == "D")
    w = power_match_weights(c[sd], mu1[idx][sd], mu2[idx][sd], np.zeros(sd.sum()), lv[sd])
    dw = (c == "D") | (c == "W")
    real = (lv >= 1) & (lv <= 4)
    sdr = sd & real
    wr = power_match_weights(c[sdr], mu1[idx][sdr], mu2[idx][sdr], np.zeros(sdr.sum()), lv[sdr])
    return np.array([average_precision(c[sd] == "S", s[sd], w), roc_auc(c[dw] == "D", PD[dw]),
                     average_precision(c[sdr] == "S", s[sdr], wr)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-set", default="valid_dev")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    _, meta = load_all(a.valid_set, None)
    m = meta.numpy()
    cls = np.array(CLASS_ORDER)[m[:, MI["y"]].astype(int)]
    mu1, mu2, level = m[:, MI["mu1"]], m[:, MI["mu2"]], m[:, MI["level"]].astype(int)
    P = {r: np.load(OUT / r / "valid_probs.npy") for r in a.runs}
    n = len(cls)
    for r in a.runs:
        assert len(P[r]) == n, (r, len(P[r]), n)
    full = {r: endpoints(P[r], cls, mu1, mu2, level, np.arange(n)) for r in a.runs}
    g = np.random.default_rng(20261029)
    boots = {r: [] for r in a.runs}
    for _ in range(a.n_boot):
        idx = g.integers(0, n, n)
        for r in a.runs:
            boots[r].append(endpoints(P[r], cls, mu1, mu2, level, idx))
    names = ["auprc_sd_pm_all", "auroc_dw_all", "auprc_sd_pm_L1-L4"]
    out = {"point": {r: dict(zip(names, full[r].round(5).tolist())) for r in a.runs}, "paired_diff": {}}
    base = a.runs[0]
    for r in a.runs[1:]:
        d = np.array(boots[base]) - np.array(boots[r])
        out["paired_diff"][f"{base} - {r}"] = {
            nm: {"diff": float(full[base][k] - full[r][k]), "ci95": np.quantile(d[:, k], [0.025, 0.975]).round(5).tolist(),
                 "p_le0": float((d[:, k] <= 0).mean())} for k, nm in enumerate(names)}
    print(json.dumps(out, indent=1))
    (OUT / f"compare_{'_vs_'.join(a.runs)}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

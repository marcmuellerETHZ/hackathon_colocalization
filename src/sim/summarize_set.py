"""Realism / composition summary of a built locus set (reads truth + stats only; no method inputs).

Reports per cell and bucket: counts, lead |z| quantiles per trait, number of causal variants, r²/d bins of D,
(mu1, mu2) quantiles, rho-hat bias / SD by class (per-window vs genome-wide mode), p, the natural class shares
of the candidate mixture (from the stats files), and generation throughput.

Usage: python -m sim.summarize_set --set screen_v1 [--dir data/processed/sets] [--out src/sim/validation]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.truth import read_truth
from sim import params

ROOT = Path(__file__).resolve().parents[2]


def q(x, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {f"q{int(100 * k)}": float(np.quantile(x, k)) for k in qs} if len(x) else {}


def summarize(set_name: str, d: Path) -> dict:
    t = read_truth(d / f"{set_name}.truth.parquet")
    out = {"set": set_name, "n_loci": int(len(t)), "cells": {}}
    t["n_causal1"] = t["causal1"].map(len)
    t["n_causal2"] = t["causal2"].map(len)
    for (cell, bucket), g in t.groupby(["cell_id", "bucket"]):
        r = {
            "n": int(len(g)),
            "lead_abs_z1": q(g["lead_abs_z1"]), "lead_abs_z2": q(g["lead_abs_z2"]),
            "mu1_cd": q(g["mu1"]), "mu2_cd": q(g["mu2"]),
            "n_causal1": g["n_causal1"].value_counts().sort_index().to_dict(),
            "n_causal2": g["n_causal2"].value_counts().sort_index().to_dict(),
            "locus_type": g["locus_type"].value_counts().to_dict(),
            "background_share": float(g["background"].mean()),
            "p": q(g["p"]),
            "N1": g["N1"].value_counts().sort_index().to_dict(), "N2": g["N2"].value_counts().sort_index().to_dict(),
            "pairing_sub": g["pairing_sub"].value_counts().to_dict(),
            "rho_true": g["rho"].value_counts().to_dict(),
            "rho_hat_minus_rho": {"mean": float((g["rho_hat"] - g["rho"]).mean()),
                                  "sd": float((g["rho_hat"] - g["rho"]).std()),
                                  "mean_se": float(g["rho_se"].mean())},
            "rho_mode": g["rho_mode"].value_counts().to_dict(),
        }
        if bucket.startswith("D"):
            r2e = [b[0] for b in params.R2_BINS] + [params.R2_BINS[-1][1]]
            de = [b[0] for b in params.D_BINS] + [np.inf]
            r["r2_bins"] = pd.cut(g["r2"], r2e, right=False).value_counts().sort_index().astype(int).rename(str).to_dict()
            r["d_bins"] = pd.cut(g["d"], de, right=False).value_counts().sort_index().astype(int).rename(str).to_dict()
        out["cells"].setdefault(cell, {})[bucket] = r
    # rho-hat by mode and class (pooled)
    rh = {}
    for (mode, cls), g in t.groupby(["rho_mode", "oracle_class"]):
        rh[f"{mode}|{cls}"] = {"n": int(len(g)), "bias": float((g["rho_hat"] - g["rho"]).mean()),
                               "bias_se": float((g["rho_hat"] - g["rho"]).std() / np.sqrt(len(g))),
                               "sd": float((g["rho_hat"] - g["rho"]).std()), "mean_se": float(g["rho_se"].mean())}
    out["rho_hat_by_mode_class"] = rh
    stats = {}
    for f in sorted(d.glob(f"{set_name}.stats*.json")):
        s = json.loads(f.read_text())
        stats[f.name] = s
    mix, rt = {}, {}
    for fname, s in stats.items():
        for cid, c in s.get("cells", {}).items():
            n = c["n_candidates"]
            mix[cid] = {"n_candidates": n, "n_none": c["n_none"],
                        "class_shares": {k: v / max(1, n - c["n_none"]) for k, v in sorted(c["candidate_class_counts"].items())},
                        "bucket_acceptance": {k: v / max(1, n) for k, v in sorted(c["candidate_bucket_counts"].items())},
                        "t_candidates_s": c["t_candidates_s"], "t_finalize_write_s": c["t_finalize_write_s"],
                        "candidates_per_s": n / max(1e-9, c["t_candidates_s"])}
        rt[fname] = {"runtime_s": s.get("runtime_s"), "visits": s.get("visits")}
    out["candidate_mixture"] = mix
    out["runtime"] = rt
    tot_rt = max([v["runtime_s"] or 0 for v in rt.values()] or [0])
    out["throughput"] = {"loci": int(len(t)), "wall_s_max_process": tot_rt,
                         "loci_per_s_wall": len(t) / tot_rt if tot_rt else None}
    return out


def to_md(s: dict) -> str:
    L = [f"# Set {s['set']}: {s['n_loci']} loci", "", "## Natural class shares of the candidate mixture (per cell)", "",
         "| cell | candidates | S | D | W | G | U | acceptance S / D<.5 / D.5-.8 / W | cand/s |", "|---|---|---|---|---|---|---|---|---|"]
    for cid, m in sorted(s["candidate_mixture"].items()):
        cs, ba = m["class_shares"], m["bucket_acceptance"]
        L.append(f"| {cid} | {m['n_candidates']} | " + " | ".join(f"{cs.get(k, 0):.3f}" for k in "SDWGU") +
                 f" | {ba.get('S', 0):.3f} / {ba.get('D_r2lt0.5', 0):.3f} / {ba.get('D_r2_0.5-0.8', 0):.4f} / {ba.get('W', 0):.3f}"
                 f" | {m['candidates_per_s']:.0f} |")
    L += ["", "## Lead |z| (median [5%, 95%]) and causal counts per bucket", "",
          "| cell | bucket | n | lead|z1| | lead|z2| | #causal1 | #causal2 | p (median) | rho_hat - rho (mean ± sd) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for cid, bk in sorted(s["cells"].items()):
        for b, r in sorted(bk.items()):
            z1, z2 = r["lead_abs_z1"], r["lead_abs_z2"]
            L.append(f"| {cid} | {b} | {r['n']} | {z1['q50']:.1f} [{z1['q5']:.1f}, {z1['q95']:.1f}] | "
                     f"{z2['q50']:.1f} [{z2['q5']:.1f}, {z2['q95']:.1f}] | {r['n_causal1']} | {r['n_causal2']} | "
                     f"{r['p']['q50']:.0f} | {r['rho_hat_minus_rho']['mean']:+.3f} ± {r['rho_hat_minus_rho']['sd']:.3f} |")
    L += ["", "## D strata (r² bins / d bins)", ""]
    for cid, bk in sorted(s["cells"].items()):
        for b, r in sorted(bk.items()):
            if "r2_bins" in r:
                L.append(f"- {cid} {b}: r² {r['r2_bins']}; d {r['d_bins']}")
    L += ["", "## rho-hat by mode and oracle class", "", "| mode / class | n | bias (SE) | sd | mean reported SE |",
          "|---|---|---|---|---|"]
    for k, r in sorted(s["rho_hat_by_mode_class"].items()):
        L.append(f"| {k} | {r['n']} | {r['bias']:+.4f} ({r['bias_se']:.4f}) | {r['sd']:.3f} | {r['mean_se']:.3f} |")
    L += ["", f"Throughput: {s['throughput']}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="screen_v1")
    ap.add_argument("--dir", default=str(ROOT / "data/processed/sets"))
    ap.add_argument("--out", default=str(ROOT / "src/sim/validation"))
    a = ap.parse_args()
    s = summarize(a.set, Path(a.dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.set}_summary.json").write_text(json.dumps(s, indent=1, default=str))
    (out / f"{a.set}_summary.md").write_text(to_md(s))
    print(to_md(s))


if __name__ == "__main__":
    main()

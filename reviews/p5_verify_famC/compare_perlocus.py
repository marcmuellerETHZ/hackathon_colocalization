"""p5.verify_famC_scoring: per-locus comparison of the independent recompute (raw logits, own softmax/KL) with the
c1_eval.py rerun (job p5_verify_famC_recompute_r2) on the pilot set, c1_long 8 M; plus a prior-consistency check:
natural-weighted mean posterior of B, C (no correction) and C (old correction) vs the natural class shares."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/cluster/scratch/mumarc/hackathon_eqtl_gwas/results/p5_verify_famC")
CL = ["S", "D", "W", "G", "U"]
me = json.loads((R / "recompute_c1_long8M_pilot.json").read_text())
L = pd.DataFrame(me["loci"]).set_index("locus_id")
out = {}
for m in ("none", "main"):
    ev = pd.read_parquet(R / f"c1_eval_pilot_{m}.model.preds.parquet").set_index("locus_id").loc[L.index, CL]
    mine = np.stack(L["PC_none" if m == "none" else "PC_old"].to_numpy())
    d = np.abs(mine - ev.to_numpy(float))
    det = L["detail"].to_numpy()
    out[m] = {"max_abs_dP_all600": float(d.max()), "max_abs_dP_detail50": float(d[det].max())}
# per-locus KL vs c1_eval (recomputed from c1_eval's stored preds with c1_eval's own kl_rows)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "famC_eval"))
from c1_eval import kl_rows  # noqa: E402
PB = np.stack(L["PB"].to_numpy())
for m, col in (("none", "KL_none"), ("main", "KL_old")):
    ev = pd.read_parquet(R / f"c1_eval_pilot_{m}.model.preds.parquet").set_index("locus_id").loc[L.index, CL]
    k_ev = kl_rows(PB, ev.to_numpy(float))
    det = L["detail"].to_numpy()
    out[m]["max_abs_dKL_detail50"] = float(np.abs(k_ev[det] - L[col].to_numpy()[det]).max())
    out[m]["max_abs_dKL_all600"] = float(np.abs(k_ev - L[col].to_numpy()).max())
# natural-weighted mean posteriors
nat = me["natural_shares_stats"]
cls = L["cls"].to_numpy()
share = pd.Series(cls).value_counts(normalize=True)
z = sum(nat[c] for c in ("S", "D", "W", "G"))
w = np.array([nat[c] / z / share[c] for c in cls])
w /= w.sum()
mp = {}
for nm, col in (("B", "PB"), ("C_none", "PC_none"), ("C_old", "PC_old")):
    P = np.stack(L[col].to_numpy())
    mp[nm] = dict(zip(CL, (w @ P).round(4).tolist()))
mp["natural_shares_renorm_SDWG"] = {c: round(nat[c] / z, 4) for c in ("S", "D", "W", "G")}
out["natural_weighted_mean_posterior"] = mp
# 10 example loci (detail subset): class, B, C_none, C_old, KLs
ex = L[L["detail"]].sort_values("KL_none", ascending=False).head(6)
out["examples_top6_by_KL_none"] = [
    {"locus_id": i, "cls": r["cls"], "PB": np.round(r["PB"], 3).tolist(), "PC_none": np.round(r["PC_none"], 3).tolist(),
     "PC_old": np.round(r["PC_old"], 3).tolist(), "KL_none": round(r["KL_none"], 4), "KL_old": round(r["KL_old"], 4)}
    for i, r in ex.iterrows()]
print(json.dumps(out, indent=1))
Path(__file__).with_name("compare_perlocus.out.json").write_text(json.dumps(out, indent=1))

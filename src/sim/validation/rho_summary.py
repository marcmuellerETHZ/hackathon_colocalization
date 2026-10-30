"""Summarise the rho-hat calibration runs into validation/rho_calibration.md (final estimator per level:
residualised + masked + pruned, jackknife-10 SE at L0/L1; masked + pruned, jackknife-10 at L2)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
a = pd.read_parquet(D / "rho_calibration_resid.parquet")
b = pd.read_parquet(D / "rho_calibration_resid_L1.parquet")
df = pd.concat([a, b], ignore_index=True)
final = np.where(df["level"].isin(["L0", "L1"]), df["rho_r_p_m_1p_jk10"], df["rho_p_m_1p_jk10"])
final_se = np.where(df["level"].isin(["L0", "L1"]), df["se_r_p_m_1p_jk10"], df["se_p_m_1p_jk10"])
df["rho_final"], df["se_final"] = final, final_se
rows = []
for (c, L, t), s in df.groupby(["config", "level", "type"]):
    rec = {"config": c, "level": L, "type": t, "n": len(s)}
    for tag in ("final", "old_1p_info"):
        d = (s[f"rho_{tag}"] - s["rho"]).to_numpy(float)
        se = s[f"se_{tag}"].to_numpy(float)
        ok = np.isfinite(d) & np.isfinite(se)
        d, se = d[ok], se[ok]
        rec[f"{tag}_bias"] = d.mean()
        rec[f"{tag}_bias_mcse"] = d.std() / np.sqrt(len(d))
        rec[f"{tag}_sd"] = d.std()
        rec[f"{tag}_ratio"] = d.std() / np.sqrt((se ** 2).mean())
    rows.append(rec)
t = pd.DataFrame(rows)
gw = json.loads((D / "rho_genomewide_calibration.json").read_text())
lines = ["# rho-hat calibration (validation split; src/sim/validate_rho.py, README D14)", "",
         "Final estimator: D6(d) signal mask (r² > 0.1 around max|z| > 4) + DESIGN §2.4 pruning (r² < 0.1) + |z| < 2,",
         "unit-variance truncated MLE, SE = delete-one-block jackknife (10 blocks); strong signals residualised out of",
         "both traits when the LD is accurate (L0 in-sample, L1 n_ref = 2,000). `old` = M3 estimator (pruning, info SE).",
         "ratio = SD(rho_hat - rho) / RMS(reported SE); calibrated if 0.85-1.15. Types: N null, S shared (random",
         "signs), Sc shared with concordant signs, D distinct, W one trait. 200-250 loci per row.", "",
         "| config | level | type | n | bias (final) | SD | ratio | bias (old) | SD | ratio |", "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in t.iterrows():
    lines.append(f"| {r.config} | {r.level} | {r.type} | {r.n} | {r.final_bias:+.3f} ± {r.final_bias_mcse:.3f} | "
                 f"{r.final_sd:.3f} | {r.final_ratio:.2f} | {r.old_1p_info_bias:+.3f} | {r.old_1p_info_sd:.3f} | "
                 f"{r.old_1p_info_ratio:.2f} |")
lines += ["", "Genome-wide mode (20,000 independent SNPs, 1% polygenic; 400 replicates, jackknife-10 SE):", "",
          "| rho | bias | SD | ratio |", "|---|---|---|---|"]
for k, v in gw.items():
    lines.append(f"| {k} | {v['bias']:+.4f} | {v['sd']:.4f} | {v['ratio']:.2f} |")
(D / "rho_calibration.md").write_text("\n".join(lines) + "\n")
t.to_csv(D / "rho_calibration_final.tsv", sep="\t", index=False)
print("\n".join(lines))

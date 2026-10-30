"""End-to-end checks of the extracted real locus sets (run after extract.py).

Checks (each reported; hard failures raise):
  1. every locus validates (io.LocusData.validate), split in {train, valid}, window inside its gene's
     genes_v1 window, all positions inside the window, variant ids unique and sorted by position;
  2. factor LD: diag(X'X/n) == 1 (float32 tolerance), maf_ref >= 0.01;
  3. z1/z2 equal the source beta/se (sidecar) up to the recorded allele flip: z = (+/-)beta_src/se_src;
     standardized beta/se reproduce z;
  4. harmonisation sanity: corr(study AF, pool-B AF) per trait; fraction AF-flagged;
  5. positive control (R2): at the stronger cohort's lead variant, when both |z| > 5, the two cohorts' z have
     the same sign in >= 90% of loci (allele misalignment would give ~50%);
  6. lead-variant missingness rates per arm and trait.
Output: data/processed/real/locus_sets/verify.json. Run: python -m data.real.verify
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data.real.config import MIN_MAF_REF, OUT, assert_env
from data.real.regions import gene_table

LOCUS_DIR = OUT / "locus_sets"


def verify_file(h5, sidecar: pd.DataFrame | None) -> dict:
    from eval.io import iter_loci
    genes = gene_table().set_index("gene_id")
    side = None if sidecar is None else sidecar.set_index(["locus_id", "variant_id"])
    n, bad, sign_same, sign_n = 0, [], 0, 0
    maxdiag, minmaf, zerr = 0.0, 1.0, 0.0
    lead1, lead2 = [], []
    for L in iter_loci(h5):
        n += 1
        L.validate()
        assert L.split in ("train", "valid"), (L.locus_id, L.split)
        g = genes.loc[L.extra_attrs["gene_id"]]
        assert (L.window_start, L.window_end) == (int(g["window_start"]), int(g["window_end"])), L.locus_id
        assert L.split == g["split"], L.locus_id
        assert ((L.pos > L.window_start) & (L.pos <= L.window_end)).all(), L.locus_id
        assert len(set(L.variant_id)) == L.p and np.all(np.diff(L.pos) > 0), L.locus_id
        X = np.asarray(L.X_ref, dtype=np.float64)
        d = (X * X).sum(0) / L.n_ref
        maxdiag = max(maxdiag, float(np.abs(d - 1).max()))
        minmaf = min(minmaf, float(L.maf_ref.min()))
        np.testing.assert_allclose(L.beta1 / L.se1, L.z1, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(L.beta2 / L.se2, L.z2, rtol=1e-9, atol=1e-9)
        if side is not None:
            s = side.loc[L.locus_id].loc[list(L.variant_id)]
            z1 = np.where(s["flipped1"], -1, 1) * s["beta1_src"] / s["se1_src"]
            z2 = np.where(s["flipped2"], -1, 1) * s["beta2_src"] / s["se2_src"]
            zerr = max(zerr, float(np.abs(z1 - L.z1).max()), float(np.abs(z2 - L.z2).max()))
        lead1.append(bool(L.extra_attrs["lead1_present"]))
        lead2.append(bool(L.extra_attrs["lead2_present"]))
        if L.extra_attrs["arm"] == "R2":
            j = int(np.argmax(np.maximum(np.abs(L.z1), np.abs(L.z2))))
            if abs(L.z1[j]) > 5 and abs(L.z2[j]) > 5:
                sign_n += 1
                sign_same += int(np.sign(L.z1[j]) == np.sign(L.z2[j]))
    if maxdiag > 1e-4:
        bad.append(f"LD diagonal deviates by {maxdiag}")
    if minmaf < MIN_MAF_REF - 1e-9:
        bad.append(f"maf_ref below {MIN_MAF_REF}: {minmaf}")
    if zerr > 1e-6:
        bad.append(f"z differs from source beta/se by {zerr}")
    out = {"n_loci": n, "max_abs_diag_minus_1": maxdiag, "min_maf_ref": minmaf, "max_abs_z_vs_source": zerr,
           "frac_lead1_present": float(np.mean(lead1)) if lead1 else float("nan"),
           "frac_lead2_present": float(np.mean(lead2)) if lead2 else float("nan")}
    if sign_n:
        out["R2_sign_concordance"] = {"n": sign_n, "frac_same_sign": sign_same / sign_n}
        if sign_same / sign_n < 0.9:
            bad.append(f"R2 sign concordance {sign_same / sign_n:.3f} < 0.9")
    if sidecar is not None and len(sidecar):
        for t in ("1", "2"):
            a = sidecar[f"af{t}"].to_numpy(float)
            ok = np.isfinite(a)
            out[f"corr_af{t}_poolB"] = float(np.corrcoef(a[ok], sidecar["af_poolB"].to_numpy(float)[ok])[0, 1])
            out[f"frac_af_flag{t}"] = float(sidecar[f"af_flag{t}"].mean())
            out[f"frac_flipped{t}"] = float(sidecar[f"flipped{t}"].mean())
    out["failures"] = bad
    return out


def main() -> None:
    assert_env()
    res = {}
    for h5 in sorted(LOCUS_DIR.glob("*.h5")):
        stem = h5.name.split(".")[0]
        sc = LOCUS_DIR / f"{stem}.variants.parquet"
        sidecar = pd.read_parquet(sc) if sc.exists() else None
        res[h5.name] = verify_file(h5, sidecar)
        print(h5.name, json.dumps(res[h5.name]), flush=True)
    (LOCUS_DIR / "verify.json").write_text(json.dumps(res, indent=1))
    fails = {k: v["failures"] for k, v in res.items() if v["failures"]}
    if fails:
        raise SystemExit(f"verification failures: {fails}")


if __name__ == "__main__":
    main()

"""Independent recount of G-mm identity (DESIGN_P2 §2; DECISIONS P4-3 strict, P4-3a amended, P4-6).
G-vs-R deltas (max|dPIP|, G status/niter, G maxH4) come from cmp_v7 (G PIP vectors are not stored);
R status, R PIPs, R PP.H4 and the 10-seed R-vs-R spreads are recomputed here from the raw R JSONs."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
D = Path("/cluster/scratch/mumarc/hackathon_eqtl_gwas/results/famG_mm/identity_v2")
fits = pd.read_csv(D / "cmp_v7/fits_float64_cuda.tsv", sep="\t")
pairs = pd.read_csv(D / "cmp_v7/pairs_float64_cuda.tsv", sep="\t").set_index("locus_id")
ids = (D / "ids.txt").read_text().split()
assert len(ids) == 100 and len(fits) == 200 and set(fits.locus_id) == set(ids)
def h4(js):
    pr = js.get("pairs")
    return None if pr is None else float(np.asarray(pr["PP"], float).reshape(-1, 5)[:, 4].max())
sys.path.insert(0, "/cluster/scratch/mumarc/hackathon_eqtl_gwas/src")
from baselines.abf import coloc_abf
from eval.io import LocusSetReader
def abf(lid):
    with LocusSetReader(f"/cluster/scratch/mumarc/hackathon_eqtl_gwas/data/processed/sets/valid_grid_v1/{lid.split('__')[1]}.h5") as rd:
        lc = rd.read(lid, load_ld=False)
    return float(coloc_abf(lc.z1, lc.se1, lc.z2, lc.se2, W=0.15, p12=5e-6)[4])
R0 = {i: json.load(open(D / "r016" / f"{i}.json")) for i in ids}
# --- R status from raw JSON vs TSV; status agreement
rconv = np.array([bool(R0[l][f"t{t}"]["converged"]) for l, t in zip(fits.locus_id, fits.trait)])
rerr = np.array([R0[l][f"t{t}"]["error"] is not None for l, t in zip(fits.locus_id, fits.trait)])
rnit = np.array([int(R0[l][f"t{t}"]["niter"]) for l, t in zip(fits.locus_id, fits.trait)])
assert (rconv == fits.conv_r.values).all() and (rnit == fits.niter_r.values).all() and (rerr == fits.r_failed.values).all()
st = (fits.conv_r == fits.conv_g) & (fits.r_failed == fits.g_failed) & (fits.niter_r == fits.niter_g)
print(f"R: {(~rconv).sum()} non-converged, {rerr.sum()} failed; status(conv,fail,niter) agree {st.sum()}/200")
conv = fits[fits.conv_r]
print(f"converged-in-R fits {len(conv)}: max|dPIP| {conv.max_dpip.max():.3e}; n<1e-3 {(conv.max_dpip<1e-3).sum()}")
# PP.H4 on loci with both traits converged in R
bothc = [l for l in ids if R0[l]["t1"]["converged"] and R0[l]["t2"]["converged"]]
dh = []
for l in bothc:
    pr = pairs.loc[l]; assert bool(pr.r_pairs) == (R0[l].get("pairs") is not None)
    if pr.r_pairs and pr.g_pairs: dh.append(abs(pr.maxH4_g - h4(R0[l])))
    elif pr.r_pairs != pr.g_pairs: dh.append(np.inf)
    else: dh.append(0.0)   # no pair in either: same coloc.abf fallback on identical z
print(f"both-converged loci {len(bothc)}: max|dPP.H4| {max(dh):.3e}; pair presence agree all 100: {(pairs.r_pairs==pairs.g_pairs).all()}")
# --- non-converged fits: spreads from raw perturbed R JSONs
rows = []
for l in ids:
    Rs = [json.load(open(D / "pert" / f"s{s}" / f"{l}.json")) for s in range(1, 11)] if (D/"pert/s1"/f"{l}.json").exists() else None
    for t in (1, 2):
        r0 = R0[l][f"t{t}"]
        if r0["converged"]: continue
        assert Rs is not None
        sp_pip = max(np.abs(np.asarray(r[f"t{t}"]["pip"]) - np.asarray(r0["pip"])).max() for r in Rs)
        h0 = h4(R0[l]); hs = [h4(r) for r in Rs]; pr = pairs.loc[l]
        if any(h is None for h in [h0] + hs) or not pr.g_pairs:   # wrapper's coloc.abf fallback (W=0.15, p12=5e-6)
            fb = abf(l); h0 = fb if h0 is None else h0; hs = [fb if h is None else h for h in hs]
            hg = pr.maxH4_g if pr.g_pairs else fb
        else: hg = pr.maxH4_g
        sp_h4 = max(abs(h - h0) for h in hs); dH4 = abs(hg - h0)
        g = fits[(fits.locus_id == l) & (fits.trait == t)].iloc[0]
        a = (not g.conv_g) and int(g.niter_g) == int(r0["niter"]) and not g.g_failed
        rows.append(dict(fit=f"{l.split('__',1)[1]} t{t}", a=a, dpip=g.max_dpip, sp_pip=sp_pip, dH4=dH4, sp_h4=sp_h4,
                         strict=a and g.max_dpip <= sp_pip and dH4 <= sp_h4,
                         amended=a and g.max_dpip <= max(sp_pip, 1e-3) and dH4 <= max(sp_h4, 1e-3)))
df = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(df.to_string(float_format=lambda x: f"{x:.3g}"))
print(f"non-converged fits {len(df)}: (a) {df.a.sum()}, strict P4-3 {df.strict.sum()}/{len(df)}, amended P4-3a {df.amended.sum()}/{len(df)}")
print("amended failures:", df[~df.amended].fit.tolist()); print("strict failures:", df[~df.strict].fit.tolist())
# cross-check vs stored sensitivity.tsv
s = pd.read_csv(D / "sens_v7/sensitivity.tsv", sep="\t")
print("stored sens_v7: strict", s.match.sum(), "amended", s.match_or_1e3.sum(),
      "| max |spread_pip diff|", np.abs(np.sort(s.spread_pip.values) - np.sort(df.sp_pip.values)).max())
ok = st.all() and conv.max_dpip.max() < 1e-3 and max(dh) < 1e-3 and df.amended.all()
print("VERDICT correctness rung (P4-3a as written):", "PASS" if ok else "FAIL")

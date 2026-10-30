"""Spot check on one converged identity locus (CPU only, CUDA hidden):
 (1) real coloc_susie_mm wrapper (baselines.methods.ColocSusieMM, fresh R run) vs
 (2) G-mm Method famG.method_mm.GPUColocSusieMM(ld_dtype=float64, device=cpu, use_triton=False) on the same harness locus."""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/cluster/scratch/mumarc/hackathon_eqtl_gwas/src")
import numpy as np
from eval.io import LocusSetReader
from baselines.methods import ColocSusieMM
from famG.method_mm import GPUColocSusieMM
lid = sys.argv[1]
with LocusSetReader(f"/cluster/scratch/mumarc/hackathon_eqtl_gwas/data/processed/sets/valid_grid_v1/{lid.split('__')[1]}.h5") as rd:
    lc = rd.read(lid)
t = time.time(); r = ColocSusieMM().fit_one(lc); tr = time.time() - t
t = time.time(); g = GPUColocSusieMM(ld_dtype="float64", device="cpu", use_triton=False).fit_batch([lc])[0]; tg = time.time() - t
print(f"R wrapper {tr:.1f}s status={r['status']} | G-mm cpu {tg:.1f}s status={g['status']}")
for k in ("s", "P_S", "P_D", "P_W"):
    print(f"{k}: R {r[k]!r}  G {g[k]!r}  |d| {abs(r[k]-g[k]):.3e}")
er, eg = json.loads(r["extra"]) if isinstance(r["extra"], str) else r["extra"], json.loads(g["extra"])
print("R extra:", {k: v for k, v in er.items() if k in ("niter", "converged", "n_cs1", "n_cs2", "pairs_PP")})
print("G extra:", {k: v for k, v in eg.items() if k in ("niter", "converged", "n_cs1", "n_cs2", "pairs_PP")})

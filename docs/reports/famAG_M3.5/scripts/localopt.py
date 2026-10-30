"""Local-optimum check on smoke D loci: default IBSS vs warm start with the shared state switched off for the
first 5 sweeps (same final model: fixed pi), compare final ELBO and P_S/P_D."""
import math, numpy as np, pandas as pd, torch
from eval.io import LocusSetReader
from famA import joint_susie as js
from famA.method import locus_types
df = pd.read_csv("reports/famAG_M3.5/smoke_screen_v1.tsv", sep="\t")
d = df[(df.run=="A_rho_eta_pifix")&(df.oracle_class.isin(["D","S"]))]
with LocusSetReader("data/processed/sets/screen_v1.h5") as rd: loci=[rd.read(i) for i in d.locus_id]
cfg = js.JointConfig(pi_mode="fixed")
def fit(lc, warm):
    kind = lc.ld_type; ld = lc.X_ref if kind=="factor" else lc.R
    it = dict(z1=lc.z1, z2=lc.z2, ld=ld, n=(float(np.median(lc.n1)), float(np.median(lc.n2))), rho=float(lc.rho_hat) if math.isfinite(lc.rho_hat) else 0.0,
              n_ref=lc.n_ref, thr=js.thresholds_for_types([locus_types(lc)[0]], cfg)[0], nref_e=None, maf=None)
    st = js._make_state([it], kind, lc.p, cfg, "cuda", n_pad=(ld.shape[0] if kind=="factor" else None))
    lp = st.logpi.clone()
    if warm:
        st.logpi[:, 3] = -math.inf; st.logpi[:, 1:3] = math.log(0.375); 
    prev = -math.inf
    for i in range(100):
        if warm and i == 5: st.logpi = lp.clone(); prev = -math.inf
        st.sweep(); el = float(st.elbo())
        if i >= (5 if warm else 0) and el - prev < 1e-3: break
        prev = el
    o = st.outputs([0])[0]
    return el, o["P_S"], o["P_D"]
rows=[]
for lc, cls in zip(loci, d.oracle_class):
    a = fit(lc, False); b = fit(lc, True)
    rows.append(dict(cls=cls, cell=lc.cell_id, elbo_default=a[0], elbo_warm=b[0], PS_d=a[1], PD_d=a[2], PS_w=b[1], PD_w=b[2]))
r = pd.DataFrame(rows); r["delbo"] = r.elbo_warm - r.elbo_default
pd.set_option("display.width", 200)
print(r.round(3).to_string())
for c in ("D","S"):
    x = r[r.cls==c]; print(c, "n", len(x), "warm ELBO higher by >1:", int((x.delbo>1).sum()), "lower by >1:", int((x.delbo<-1).sum()),
          "median PD default %.3f warm %.3f; PS default %.3f warm %.3f" % (x.PD_d.median(), x.PD_w.median(), x.PS_d.median(), x.PS_w.median()))
r.to_csv("reports/famAG_M3.5/smoke_local_optimum.tsv", sep="\t", index=False)

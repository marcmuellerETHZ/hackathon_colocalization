"""Engineering check: do coloc-SuSiE loci that did not converge at the susie_rss default max_iter converge with
more iterations (coloc::runsusie re-runs with 100x max_iter), and do the scores change?
Usage: python src/baselines/conv_check.py <set.h5> <pilot runs.json> <out.tsv> 1000,10000
(record of the screen_v1 dry run: benchmarks/convergence_screen_v1.tsv; README decision 19)
"""
import json, sys, time
from multiprocessing import get_context
import pandas as pd

SET = sys.argv[1]; RUNS = sys.argv[2]; OUT = sys.argv[3]; MAXIT = [int(x) for x in sys.argv[4].split(",")]

def work(arg):
    m, lid = arg
    from eval.io import read_locus
    from baselines.methods import ColocSusie
    l = read_locus(SET, lid)
    out = []
    for mi in MAXIT:
        meth = ColocSusie(mismatch=(m == "coloc_susie_mm"), max_iter=mi, timeout=3000)
        r = meth.fit_one(l)
        e = json.loads(r["extra"])
        out.append(dict(method=m, locus_id=lid, max_iter=mi, status=r["status"], s=r["s"], P_S=r["P_S"], P_D=r["P_D"],
                        runtime=r["runtime_s"], fallback=e.get("fallback"),
                        niter1=(e.get("t1") or {}).get("niter"), niter2=(e.get("t2") or {}).get("niter"),
                        ncs1=(e.get("t1") or {}).get("n_cs"), ncs2=(e.get("t2") or {}).get("n_cs"),
                        error=(e.get("error") or "")[:80].replace("\n", " ")))
    return out

if __name__ == "__main__":
    rec = json.load(open(RUNS))
    tasks = []
    for m in ("coloc_susie", "coloc_susie_mm"):
        p = pd.read_parquet(rec["runs"][m]["run_dir"] + "/predictions.parquet")
        tasks += [(m, lid) for lid in p.loc[p.status == "nonconverged", "locus_id"]]
    with get_context("spawn").Pool(min(len(tasks), 17)) as pool:
        res = [r for rs in pool.map(work, tasks) for r in rs]
    df = pd.DataFrame(res)
    df.to_csv(OUT, sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string())

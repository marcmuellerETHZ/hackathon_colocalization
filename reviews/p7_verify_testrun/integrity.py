#!/usr/bin/env python
"""p7 verify, checks 1 and 6: integrity of test bundle ab94c5a1 (read-only; does not import src/eval).

Usage: python reviews/p7_verify_testrun/integrity.py <project_root>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/cluster/scratch/mumarc/hackathon_eqtl_gwas")
B = ROOT / "results/eval_test/test_frozen-v1_ab94c5a1"
FROZEN = ROOT / "configs/test_methods_v1.yaml"
COMMIT = "cab2599d5ea81288468a4e7abc6d58548766edd7"
NONCE = "ab94c5a1c097449db8d1d1f4bd18c168"
KEYS = ("t_S", "t_D", "recal_a_S", "recal_a_D")
out = {}

frozen = yaml.safe_load(FROZEN.read_text())
fz = {m["name"]: m for m in frozen["methods"]}
expected_methods = sorted(set(fz) - {"mvsusie"})


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return float(a) == float(b)


# 1a. methods evaluated + metric_config
runs = sorted(p for p in (B / "runs").iterdir() if p.is_dir())
man = {p.name: json.loads((p / "manifest.json").read_text()) for p in runs}
ran = sorted({m["method_name"] for m in man.values()})
out["methods_ran"] = ran
out["methods_expected"] = expected_methods
out["methods_match"] = ran == expected_methods
mc_problems = []
for rid, m in man.items():
    name = m["method_name"]
    for k in KEYS:
        if not same(m["metric_config"].get(k), fz[name]["metric_config"][k]):
            mc_problems.append((rid, k, m["metric_config"].get(k), fz[name]["metric_config"][k]))
    if m["method"]["kwargs"] != fz[name]["kwargs"]:
        mc_problems.append((rid, "kwargs", m["method"]["kwargs"], fz[name]["kwargs"]))
opu = json.loads((B / "operating_points_used.json").read_text())
for name, op in opu.items():
    for k in KEYS:
        if not same(op.get(k), fz[name]["metric_config"][k]):
            mc_problems.append(("operating_points_used", name, k, op.get(k), fz[name]["metric_config"][k]))
out["opu_methods"] = sorted(opu)
out["metric_config_problems"] = mc_problems

# 1b. commits
commits = {}
for rid, m in man.items():
    seg_commits = {s.get("git") for s in m.get("checkpoint", {}).get("segments", [])}
    commits[rid] = (m["git"]["commit"], m["config"]["commit"], m["config"]["nonce"], sorted(seg_commits),
                    m["git"]["clean"], m.get("test_access"))
bad_commit = {r: c for r, c in commits.items()
              if c[0] != COMMIT or c[1] != COMMIT or c[2] != NONCE or c[3] != [COMMIT] or not c[4]}
out["n_runs"] = len(man)
out["runs_bad_commit_or_nonce"] = bad_commit
out["methods_files_used"] = sorted({(m["config"]["methods_file"].split("/")[-1], m["config"]["methods_file_sha256"])
                                    for m in man.values()})
out["resumed_runs"] = sorted(r for r, m in man.items() if len(m.get("checkpoint", {}).get("segments", [])) > 1)

# 1c. access log
log = [l.split("\t") for l in (ROOT / "results/test_access.log").read_text().splitlines() if l and not l.startswith("#")]
out["access_log"] = [(r[1], r[2][:7], r[3]) for r in log]
out["access_log_nonces"] = sorted({r[3] for r in log})
out["access_log_ok"] = ([r[1] for r in log] == ["started", "interrupted", "resumed", "completed"]
                        and {r[2] for r in log} == {COMMIT} and {r[3] for r in log} == {NONCE})

# 1d. units + failures per method x set (from predictions, outer-joined on the expected loci)
units = json.loads((B / "state/units.json").read_text())
out["units_total"] = len(units)
out["units_not_done"] = [k for k, v in units.items() if v.get("status") != "done"]
rows = []
for p in runs:
    m = man[p.name]
    pr = pd.read_parquet(p / "predictions.parquet", columns=["locus_id", "status", "s", "P_S", "P_D"])
    n_exp = m["counts"]["expected"]
    st = pr.status.value_counts().to_dict()
    nan_s = int(pr.s.isna().sum())
    rows.append(dict(unit=p.name.split("__", 1)[1], method=m["method_name"], set=m["config"]["set_key"],
                     expected=n_exp, rows=len(pr), ok=st.get("ok", 0), failed=st.get("failed", 0),
                     timeout=st.get("timeout", 0), nonconverged=st.get("nonconverged", 0),
                     missing=n_exp - len(pr), nan_s_on_ok=int(pr.loc[pr.status.isin(["ok", "nonconverged"]), "s"].isna().sum()),
                     manifest_counts=m["counts"]))
fails = pd.DataFrame(rows)
fails["fail_share"] = (fails.failed + fails.timeout + fails.missing) / fails.expected
fails["counts_match_manifest"] = [
    (r.ok == r.manifest_counts["ok"] and r.failed == r.manifest_counts["failed"]
     and r.timeout == r.manifest_counts["timeout"] and r.nonconverged == r.manifest_counts["nonconverged"])
    for r in fails.itertuples()]
out["failure_table"] = fails.drop(columns="manifest_counts").to_dict("records")

# per-cell failure shares (sim sets), failure = failed/timeout/missing/NaN s
cellfail = []
for setkey, truthp in [("sim_main", "testset_v1.truth.parquet"), ("sim_shifted", "testset_v1_shifted.truth.parquet")]:
    t = pd.read_parquet(ROOT / "data/processed/testset_v1/test_frozen-v1_ab94c5a1/sets" / truthp,
                        columns=["locus_id", "cell_id"])
    for p in runs:
        m = man[p.name]
        if m["config"]["set_key"] != setkey:
            continue
        pr = pd.read_parquet(p / "predictions.parquet", columns=["locus_id", "status", "s"])
        ids = pr.locus_id
        tt = t[t.locus_id.isin(ids)] if m["config"].get("subsample_per_class") or len(pr) < len(t) else t
        j = tt.merge(pr, on="locus_id", how="left")
        j["fail"] = j.status.isna() | j.status.isin(["failed", "timeout"]) | j.s.isna()
        g = j.groupby("cell_id").fail.agg(["sum", "size"])
        g["share"] = g["sum"] / g["size"]
        for c, r in g.iterrows():
            if r["share"] > 0.01:
                cellfail.append(dict(method=m["method_name"], set=setkey, cell=c, n_fail=int(r["sum"]),
                                     n=int(r["size"]), share=round(float(r["share"]), 4)))
out["cells_over_1pct"] = cellfail

# 6. runtime medians
rt = json.loads((B / "runtime.json").read_text())["rows"]
rtchk = []
for r in rt:
    rid = f"test_frozen-v1_ab94c5a1__{r['method']}__{r['set']}"
    if rid not in man:
        continue
    pr = pd.read_parquet(B / "runs" / rid / "predictions.parquet", columns=["runtime_s", "status"])
    med = float(np.median(pr.runtime_s.dropna()))
    rtchk.append(dict(unit=rid.split("__", 1)[1], runtime_json_median=r["median_s"], recomputed_median=med,
                      manifest_median=man[rid]["runtime_s"]["median"], absdiff=abs(med - r["median_s"])))
out["runtime_check"] = rtchk
out["runtime_max_absdiff"] = max(x["absdiff"] for x in rtchk)

print(json.dumps(out, indent=1, default=str))

# 1e. the resume config = frozen config minus mvsusie (parsed, ignoring free-text *_source fields and derived_from)
cut = yaml.safe_load((ROOT / "configs/test_methods_v1_d17cut1.yaml").read_text())


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not k.endswith("_source") and k not in ("derived_from", "design_ref")}
    if isinstance(d, list):
        return [strip(x) for x in d]
    return d


fz2 = strip(frozen)
fz2["methods"] = [m for m in fz2["methods"] if m["name"] != "mvsusie"]
fz2["c3_eligible_baselines"] = [m for m in fz2["c3_eligible_baselines"] if m != "mvsusie"]
print(json.dumps({"d17cut1_equals_frozen_minus_mvsusie": strip(cut) == fz2}))

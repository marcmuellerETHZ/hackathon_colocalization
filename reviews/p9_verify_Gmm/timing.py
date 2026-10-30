"""Recompute §7.7 loci/h, medians, ratios from timing_v1 (LEDGER.tsv timestamps + timing_summary.json),
and check that no other orch job's [start, end) overlapped famG_mm_timing_v1's slot in JOBS.yaml."""
import json, statistics as S
from datetime import datetime
from pathlib import Path
import pandas as pd, yaml
H = Path("/cluster/scratch/mumarc/hackathon_eqtl_gwas"); T = H / "results/famG_mm/timing_v1"
led = pd.read_csv(T / "LEDGER.tsv", sep="\t")
led["wall_ledger"] = (pd.to_datetime(led.end_utc) - pd.to_datetime(led.start_utc)).dt.total_seconds()
summ = json.load(open(T / "timing_summary.json"))
lph = {}
for m, runs in summ.items():
    lph[m] = []
    for r in runs:
        w = r["harness_wall_s"]; L = led[led.run_dir == r["run_dir"]].iloc[0]
        assert L.n_expected == 2000 and L.n_ok + L.n_nonconverged + L.n_failed == 2000 and L.n_timeout == 0
        lph[m].append(2000 / w * 3600)
        print(f"{m} rep{r['rep']} workers={r['n_workers']} wall={w:.1f}s (ledger {L.wall_ledger:.1f}s) loci/h={lph[m][-1]:.0f} "
              f"(stored {r['loci_per_hour']:.0f}) counts ok/nc/fail={L.n_ok}/{L.n_nonconverged}/{L.n_failed}")
med = {m: S.median(v) for m, v in lph.items()}
print("medians:", {m: round(v) for m, v in med.items()})
print(f"ratio G-mm/mm (medians) {med['gmm']/med['mm']:.3f}; means {S.mean(lph['gmm'])/S.mean(lph['mm']):.3f}; "
      f"pairwise range {min(lph['gmm'])/max(lph['mm']):.2f}-{max(lph['gmm'])/min(lph['mm']):.2f}")
print(f"ratio G-mm/cs rep0 only (n=1, as claimed) {med['gmm']/lph['cs'][0]:.3f}; "
      f"all complete cs reps (n={len(lph['cs'])}) median {med['cs']:.0f} -> {med['gmm']/med['cs']:.3f}")
# exclusivity
J = yaml.safe_load(open(H / "JOBS.yaml"))["jobs"]; me = [j for j in J if j["id"] == "famG_mm_timing_v1"][0]
p = lambda s: datetime.fromisoformat(s) if s else None
a, b = p(me["start"]), p(me["end"]); print("slot", me["start"], "->", me["end"], me["status"], me["failure_reason"])
ov = [(j["id"], j["status"], j.get("start"), j.get("end")) for j in J if j["id"] != me["id"] and j.get("start")
      and p(j["start"]) < b and (p(j.get("end")) or datetime.max.replace(tzinfo=a.tzinfo)) > a]
print("overlapping orch jobs:", ov or "none")

"""Figures for the manuscript. Numbers are read from the project result files where possible
(test run bundle summary.md, famC c2.json, famB m4 eval); the rest are quoted from verified reports
(cited inline). Usage: python make_figures.py <project_root> <out_dir>"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

# reference categorical palette (light mode), fixed order; text and axis tokens
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.dpi": 300, "lines.linewidth": 2,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.tif", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    from PIL import Image  # PLOS: RGB TIFF, no alpha, <= 7.5 in wide at 300 dpi
    im = Image.open(OUT / f"{name}.tif").convert("RGB")
    assert im.size[0] <= 2250, (name, im.size)
    im.save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))


# ------------------------------------------------------------------ Fig 2: AUPRC by mismatch level (test v1)
summ = (ROOT / "results/eval_test/test_frozen-v1_ab94c5a1/benchmarks/sim/summary.md").read_text().split("\n")
i = summ.index("## Per-cell AUPRC S vs D (least_favourable)")
cells = re.findall(r"(L\w+)\|P\d\|[a-z_]+=[0-9.]+", summ[i + 2])
per_level = {}
for line in summ[i + 4:]:
    if not line.startswith("|"):
        break
    cs = [c.strip() for c in line.split("|")[1:-1]]
    agg = {}
    for lv, v in zip(cells, cs[1:]):
        if v != "NA":
            agg.setdefault(lv, []).append(float(v))
    per_level[cs[0]] = {k: sum(v) / len(v) for k, v in agg.items()}
levels = ["L0", "L1", "L2", "L3", "L4", "L6"]
xlab = ["L0\nin-sample", "L1\nlarge ref.", "L2\nref. n=150", "L3\nadmixed", "L4\n+flips/missing", "L6\nFIN vs NFE"]
series = [("coloc_susie_mm", "mismatch-aware coloc-SuSiE"), ("coloc_abf", "coloc.abf"),
          ("famB_exactcoloc_gaussian_M50", "ExactColoc (B)"), ("coloc_susie", "coloc-SuSiE"),
          ("sharepro", "SharePro")]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for k, (m, lab) in enumerate(series):
    y = [per_level[m][lv] for lv in levels]
    ax.plot(range(6), y, color=S[k], marker="o", markersize=5, markeredgecolor="white", markeredgewidth=1)
    ax.annotate(lab, (5, y[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=8)
ax.set_xticks(range(6), xlab, fontsize=7.5)
ax.set_ylabel("AUPRC, shared vs distinct\n(mean over cells)")
ax.set_ylim(0.55, 1.0)
ax.set_xlim(-0.3, 5.3)
ax.grid(axis="x", visible=False)
save(fig, "Fig2_mismatch")
with open(OUT / "Fig2_mismatch_data.json", "w") as f:
    json.dump({m: per_level[m] for m, _ in series}, f, indent=1)

# ------------------------------------------------------------------ Fig 3: speed vs accuracy (test v1)
# realistic AUPRC from benchmarks/sim/summary.md (main or ColocBoost subsample); mean worker-s/locus from runtime.json
rt = json.loads((ROOT / "results/eval_test/test_frozen-v1_ab94c5a1/runtime.json").read_text())["rows"]
ws = {r["method"]: r["worker_s"] / r["n"] for r in rt if r["set"] == "sim_main"}
auprc = {"coloc_susie_mm": 0.900, "coloc_abf": 0.860, "famB_exactcoloc_gaussian_M50": 0.829, "coloc_susie": 0.717,
         "famG_coloc_susie_gpu": 0.719, "lead_snp": 0.704, "sharepro": 0.635, "coloc_prop_test": 0.632,
         "prop_coloc": 0.552, "colocboost_m1": 0.787, "colocboost": 0.678}
names = {"coloc_susie_mm": "mismatch-aware\ncoloc-SuSiE", "coloc_abf": "coloc.abf",
         "famB_exactcoloc_gaussian_M50": "ExactColoc (B)", "coloc_susie": "coloc-SuSiE",
         "famG_coloc_susie_gpu": "GPU coloc-SuSiE (G)", "lead_snp": "lead-SNP rule", "sharepro": "SharePro",
         "coloc_prop_test": "colocPropTest", "prop_coloc": "prop.coloc", "colocboost_m1": "ColocBoost m1",
         "colocboost": "ColocBoost"}
gpu = {"famB_exactcoloc_gaussian_M50", "famG_coloc_susie_gpu"}
off = {"coloc_susie_mm": (6, 4), "coloc_abf": (6, 3), "famB_exactcoloc_gaussian_M50": (6, -10),
       "coloc_susie": (-6, 10), "famG_coloc_susie_gpu": (6, 6), "lead_snp": (6, -9), "sharepro": (6, 2),
       "coloc_prop_test": (-6, -11), "prop_coloc": (6, -3), "colocboost_m1": (6, 2), "colocboost": (6, -10)}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for m, a in auprc.items():
    g = m in gpu
    ax.scatter(ws[m], a, s=46, marker="s" if g else "o", color=S[1] if g else S[0], edgecolor="white",
               linewidth=1.2, zorder=3)
    ax.annotate(names[m], (ws[m], a), xytext=off[m], textcoords="offset points", fontsize=7.5, color=INK, va="center",
                ha="right" if off[m][0] < 0 else "left")
ax.set_xscale("log")
ax.set_xlabel("Mean compute per locus (worker-seconds; 1 CPU core or 1 GPU), log scale")
ax.set_ylabel("Realistic AUPRC, shared vs distinct")
ax.set_ylim(0.52, 0.93)
ax.scatter([], [], marker="o", color=S[0], label="CPU, 1 core per worker")
ax.scatter([], [], marker="s", color=S[1], label="GPU, batched")
ax.legend(frameon=False, loc="upper center", fontsize=8)
save(fig, "Fig3_speed_accuracy")
with open(OUT / "Fig3_speed_accuracy_data.json", "w") as f:
    json.dump({m: {"worker_s_per_locus": ws[m], "auprc_realistic": a} for m, a in auprc.items()}, f, indent=1)

# ------------------------------------------------------------------ Fig 4: phase-2 validation (M4)
c2 = json.loads((ROOT / "results/famC/c2_v1/c2.json").read_text())["endpoints_own_loci"]
bmm = json.loads((ROOT / "results/famB/p9_m4_eval/m4_Bmm_v2.json").read_text())["contrasts_block_bootstrap"]
rows = [  # (label, AUPRC, CI or None, group)   group 0 = model-based, 1 = trained on the simulator
    # CIs: reports/p9.famC_C2.md (block bootstrap); baselines: results/baselines_m4/m4_valid_v3/m4summary.md
    ("NeuralColoc, no-LD ablation", c2["nold"]["auprc_realistic"], (0.956, 0.971), 1),
    ("NeuralColoc (C, 600k)", c2["net"]["auprc_realistic"], (0.953, 0.970), 1),
    ("LightGBM floor", c2["floor"]["auprc_realistic"], (0.949, 0.966), 1),
    ("mismatch-aware coloc-SuSiE", bmm["coloc_susie_mm"]["auprc_coloc_susie_mm"], (0.887, 0.913), 0),
    ("mismatch-aware ExactColoc (B-mm)", bmm["coloc_susie_mm"]["auprc_Bmm"], None, 0),
    ("coloc.abf", 0.856, (0.842, 0.868), 0),
    ("ExactColoc (B, phase 1)", c2["B"]["auprc_realistic"], None, 0),
    ("coloc-SuSiE", 0.721, (0.700, 0.744), 0),
    ("JointSuSiE (A, phase 1)", c2["A"]["auprc_realistic"], None, 0),
]
rows.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for k, (lab, v, ci, g) in enumerate(rows):
    col = S[1] if g else S[0]
    if ci:
        ax.plot(ci, [k, k], color=col, linewidth=2, solid_capstyle="round", zorder=2)
    ax.scatter(v, k, s=46, color=col, edgecolor="white", linewidth=1.2, zorder=3, marker="s" if g else "o")
    if k == 3:  # coloc.abf: label above the point, clear of the dashed reference line
        ax.annotate(f"{v:.3f}", (v, k), xytext=(0, 9), textcoords="offset points", ha="center", va="center", fontsize=7.5, color=INK2)
    else:
        ax.annotate(f"{v:.3f}", (ci[1] if ci else v, k), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
ax.axvline(bmm["coloc_susie_mm"]["auprc_coloc_susie_mm"], color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
ax.set_xlabel("Realistic AUPRC, shared vs distinct (validation subsample M4)")
ax.set_xlim(0.64, 1.01)
ax.grid(axis="y", visible=False)
ax.scatter([], [], marker="o", color=S[0], label="model-based")
ax.scatter([], [], marker="s", color=S[1], label="trained on the same simulator")
ax.legend(frameon=False, loc="upper left", fontsize=8)
save(fig, "Fig4_phase2_validation")


# ------------------------------------------------------------------ Fig 1: agent setup and screening flow
def box(ax, x, y, w, h, text, fc="#f1f0ec", ec=INK2, fs=6.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))


fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
# left: agents
box(ax, 0.02, 0.80, 0.30, 0.15, "PI agent (one session at a time)\nplans, decides, merges, writes\nRESEARCH_TREE / DECISIONS / LESSONS", bold=False)
box(ax, 0.02, 0.56, 0.30, 0.17, "typically ≤ 6 sub-agents, own worktrees\ndata · simulator · evaluation\nbaselines · families A/B/C/G\nreviewers · verifiers · retrospectives")
box(ax, 0.02, 0.33, 0.30, 0.16, "Job queue (orch)\ncode snapshots, declared CPU/RAM/VRAM\nJOBS / TASKS / ARTIFACTS registries\nreconcile + heartbeat")
arrow(ax, 0.17, 0.80, 0.17, 0.73)
arrow(ax, 0.17, 0.56, 0.17, 0.49)
ax.text(0.17, 0.27, "Independent checks: kill verdicts and\nheadline numbers recomputed by an agent\nthat did not produce them (A1 only at the end)", ha="center", va="top", fontsize=6.4, color=INK2)
# right: pipeline
xs = 0.40
steps = [
    (0.87, "Literature (3 agents) → design v1–v4, 3 review rounds, frozen"),
    (0.73, "Simulator (1000G 30x, Li–Stephens mosaics)\nLD-block splits 60/20/20 · real-data arms"),
    (0.59, "Screening kill tests on validation data\nA1 ✗ · B1 ✗ · C0 ✓ · C1 ✓ · C2 not run"),
    (0.45, "No candidate → pre-registered fallback:\nbenchmark of existing methods"),
    (0.31, "Single test run (v1: 13,207 loci + real arms)\nindependently verified"),
    (0.17, "Phase 2 (validation only, new pre-registration): B-mm ✗ · G-mm ✗ · C2 ✗\n→ test set v2 never generated"),
]
for y, t in steps:
    box(ax, xs, y, 0.56, 0.10, t, fc="#eef4fc" if "Phase 2" not in t else "#fdf0ea")
for (y0, _), (y1, t1) in zip(steps[:-1], steps[1:]):
    arrow(ax, xs + 0.28, y0, xs + 0.28, y1 + 0.10)
ax.text(xs, 0.06, "Phase 1: 10-26 ≈13:19 → 10-28 ≈05:06    Phase 2: 10-28 ≈09:30 → ≈18:45 (git times)", fontsize=7.2, color=INK2)
save(fig, "Fig1_overview")
print("figures written to", OUT)

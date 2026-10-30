"""Paths, dataset tables and fixed constants for the real-data arms (DESIGN §2.5). Owner: real-data engineer.

Everything that is a *choice* (not fixed by DESIGN) is documented in src/data/real/README.md,
"Decisions / assumptions".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW_EQTL = ROOT / "data/raw/eqtlcat"
RAW_FG = ROOT / "data/raw/finngen/r12"
OUT = ROOT / "data/processed/real"
SCRIPTS = ROOT / "scripts/real"
SPLITS = ROOT / "datasets" / "splits"

FINNGEN_LD_URL = "https://storage.googleapis.com/finngen-public-data-r12/ld_matrix/finngen_r12_chr{c}_ld.tsv.gz"

# ---- observed power gate (DESIGN §1) ----
ALPHA_GWAS = 5e-8      # trait 1 (FinnGen)
ALPHA_QTL = 1e-5       # trait 2 (eQTL); also used for BOTH cohorts in R2 (eQTL x eQTL)
CS_MIN_R2 = 0.25       # purity >= 0.5 as min |r|  <=>  min r^2 >= 0.25 (eQTL Catalogue cs_min_r2)
R3B_MAX_LEAD_R2 = 0.1  # R3b: lead r^2 < 0.1 in pool B and FinnGen LD
MIN_MAF_REF = 0.01     # variant set: MAF >= 0.01 in the reference panel (DESIGN §2.1)
HALF_WINDOW = 500_000  # TSS +- 500 kb (DESIGN §3), clipped to same-split block run

# ---- harmonisation (DESIGN §2.5) ----
AF_FLAG = 0.2          # flag |dAF| > 0.2
AF_FLAG_PALINDROMIC = 0.1  # drop palindromic SNPs with |dAF| > 0.1

# pairs of datasets whose donors overlap and must never be paired in R2
SAME_DONOR_POOLS = [{"QTD000356", "QTD000266"},  # GTEx blood / liver
                    {"QTD000549", "QTD000539"}]  # TwinsUK blood / LCL


@dataclass(frozen=True)
class EqtlDataset:
    dataset_id: str
    study_id: str
    study_label: str
    sample_group: str
    cohort: str
    arms: tuple[str, ...]

    def path(self, kind: str) -> Path:
        ext = {"all": "all.tsv.gz", "permuted": "permuted.tsv.gz", "cs": "credible_sets.tsv.gz",
               "lbf": "lbf_variable.txt.gz"}[kind]
        return RAW_EQTL / "r7" / self.dataset_id / f"{self.dataset_id}.{ext}"

    @property
    def r2_group(self) -> str | None:
        g = [a.split(":", 1)[1] for a in self.arms if a.startswith("R2:")]
        return g[0] if g else None

    @property
    def label(self) -> str:
        return f"{self.study_label}_{self.sample_group}"


@dataclass(frozen=True)
class FinnGenTrait:
    pheno: str
    kind: str  # "lab" | "endpoint"
    label: str
    sumstats_prefix: str
    susie_prefix: str

    def path(self, kind: str) -> Path:
        if kind == "sumstats":
            return RAW_FG / self.sumstats_prefix / f"finngen_R12_{self.pheno}.gz"
        ext = {"cred": "SUSIE.cred.bgz", "snp": "SUSIE.snp.bgz"}[kind]
        return RAW_FG / self.susie_prefix / f"finngen_R12_{self.pheno}.{ext}"


@lru_cache(maxsize=1)
def eqtl_datasets() -> dict[str, EqtlDataset]:
    df = pd.read_csv(SCRIPTS / "eqtlcat_datasets.tsv", sep="\t", dtype=str)
    return {r.dataset_id: EqtlDataset(r.dataset_id, r.study_id, r.study_label, r.sample_group, r.cohort,
                                      tuple(r.arms.split(","))) for r in df.itertuples()}


@lru_cache(maxsize=1)
def finngen_traits() -> dict[str, FinnGenTrait]:
    df = pd.read_csv(SCRIPTS / "finngen_traits.tsv", sep="\t", dtype=str)
    return {r.pheno: FinnGenTrait(r.pheno, r.kind, r.label, r.sumstats_prefix, r.susie_prefix) for r in df.itertuples()}


@lru_cache(maxsize=1)
def finngen_sample_sizes() -> pd.DataFrame:
    """pheno, n_cases, n_controls, n_total, n_eff. Labs: Kanta documentation; endpoints: R12 manifest.
    n_eff = 4 / (1/n_cases + 1/n_controls) for endpoints, n_total for quantitative labs."""
    rows = []
    labs = pd.read_csv(RAW_FG / "lab_values/analysis_documentation/Kanta_labs_GWAS_results_v2_summary.txt",
                       sep="\t", dtype={"OMOPID": str})
    man = pd.read_csv(RAW_FG / "summary_stats/finngen_R12_manifest.tsv", sep="\t", dtype=str)
    for t in finngen_traits().values():
        if t.kind == "lab":
            r = labs[labs["OMOPID"] == t.pheno]
            assert len(r) == 1, t.pheno
            n = int(r["N_total"].iloc[0])
            rows.append((t.pheno, n, 0, n, float(n)))
        else:
            r = man[man["phenocode"] == t.pheno]
            assert len(r) == 1, t.pheno
            ca, co = int(r["num_cases"].iloc[0]), int(r["num_controls"].iloc[0])
            rows.append((t.pheno, ca, co, ca + co, 4.0 / (1.0 / ca + 1.0 / co)))
    return pd.DataFrame(rows, columns=["pheno", "n_cases", "n_controls", "n_total", "n_eff"])


def assert_env() -> None:
    if os.environ.get("COLOC_ENV_ACTIVE") != "1":
        raise RuntimeError("source env/activate.sh first")

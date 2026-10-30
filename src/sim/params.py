"""Pinned simulation parameters (DESIGN v4 §1, §2.1-2.4, decisions D6). Owner: simulation agent.

Every constant that is not fixed by DESIGN.md is documented in src/sim/README.md
("Decisions / assumptions"). Changing anything here changes the simulated world: bump WORLD_VERSION.
"""
from __future__ import annotations

from scipy.stats import norm

from common.seeds import SPLIT_SEED

WORLD_VERSION = "sim_v2"   # v2: OLS statistics (D6a), rho-hat D6(d), L0* p = 20 (D6g)

# ---------------------------------------------------------------- variant set (DESIGN §2.1)
TARGET_POOL = "A"          # MAF filter pool for the variant set
MIN_MAF = 0.01
MIN_MAF_STUDY = 0.01       # in-sample MAF filter applied to every study sample (README D3)

# ---------------------------------------------------------------- mosaic (DESIGN §2.1)
MOSAIC_SEED = SPLIT_SEED   # the mosaic world derives from the pools seed (README D1)
# Li-Stephens switch intensity = 4 * N_e / K per Morgan, K = number of source haplotypes.
# Pinned by src/sim/validate_mosaic.py (40 random 1-Mb windows genome-wide; README D2): the largest N_e on the
# grid {500, 1000, 2000, 3000, 4000, 8000} whose haplotype-r² decay is within 5% (half the DESIGN tolerance)
# of pool A in every distance bin. N_e = 3000: max |rel. err| 4.8% (src/sim/validation/mosaic_validation.json).
NE = 3000.0
# per-population N_e (same rule, validated against each population's own source pool; README D2)
# Same rule per source pool (grid {250, 500, 1000, 2000, 3000}; validation/mosaic_validation_<pool>.json):
# FIN 500 (3.2%), L6c 500 (2.9%; 1000 gives 5.3%), ASW/ACB 1000 (4.2%), pool B 1000 (3.4%).
POP_NE = {"NFE": 3000.0, "NFE_L1": 3000.0, "FIN": 500.0, "L6c": 500.0, "ADM": 1000.0,
          "SWAP": 1000.0, "SWAP_L1": 1000.0}
# miscopy (emission mismatch) probability per site and haplotype: LS default theta/(2(theta+K)) with
# Watterson theta for K = 508 is 1.4e-4; we pin that value for every pool (README D2).
THETA = 1.4e-4
N_SYN = 20_000             # synthetic NFE population size (DESIGN §2.1)
N_L1_REF = 2_000           # L1 reference: independent mosaic draw (DESIGN §2.3)
N_ADMIX_SYN = 3_000        # ASW/ACB mosaic individuals (L3 admixed component; README D6)
ADMIX_FRACTION = 0.15      # L3: 15% of the trait-2 study (DESIGN §2.3)
N_PC_SNPS = 4_000          # genome-wide SNPs for L3 PC covariates (README D6)

# populations: name -> (source pool, N individuals, seed key)
POPULATIONS = {
    "NFE": ("A", N_SYN, "NFE_main"),
    "NFE_L1": ("A", N_L1_REF, "NFE_L1ref"),
    "FIN": ("FIN", N_SYN, "FIN_main"),
    "L6c": ("L6c", N_SYN, "L6c_main"),
    "ADM": ("ASW_ACB", N_ADMIX_SYN, "ASWACB_main"),
    # pool-swap world (test-only shifted arm, DESIGN v4 §2.1/§8.6): target = mosaic of pool-B haplotypes
    "SWAP": ("B", N_SYN, "SWAP_main"),
    "SWAP_L1": ("B", N_L1_REF, "SWAP_L1ref"),
}

# ---------------------------------------------------------------- oracle (DESIGN §1, frozen)
ALPHA = {"gwas": 5e-8, "mol": 1e-5}


def _thr(alpha: float, power: float) -> float:
    """|mu| at which P(|z| > z_alpha) = power (one-sided approximation used by the design)."""
    za = norm.isf(alpha / 2)
    return float(za + norm.ppf(power))


TAU = {k: round(_thr(a, 0.8), 2) for k, a in ALPHA.items()}    # gwas 6.29, mol 5.26
WEAK = {k: round(_thr(a, 0.2), 2) for k, a in ALPHA.items()}   # gwas 4.61, mol 3.58
D_MIN = 2.0                                                      # U if d < 2

# ---------------------------------------------------------------- effects (DESIGN §2.2)
# per-variant PVE q, log-uniform; beta_std = +-sqrt(q), lambda = sqrt(N) * beta_std = sqrt(N q)
# (DESIGN v4 §1 single definition; README D4 on the q vs q² wording)
PVE_RANGE = {
    "gwas": ((1e-5, 2e-3), (2e-3, 2e-2)),   # (main stratum, biobank-strong stratum)
    "mol": ((1e-3, 0.3), None),
}
P_STRONG = 0.20            # share of the biobank-strong stratum for GWAS-like traits
MAX_TOTAL_PVE = 0.5        # resample effects if the total PVE of a trait exceeds this (R3-stat MF5)
BG_SHARE = 0.20            # loci with infinitesimal background
BG_SUM_LAMBDA2 = 2.0       # background sum lambda^2 per trait
M_MAX_CAUSAL = 3           # M loci: 2-3 causal variants per trait (DESIGN §2.4)
# shifted test arm (DESIGN v4 §8.5/§8.6): architecture overrides and locus-type mixture
SHIFTED_ARCH = (("p_strong", 0.5), ("bg_share", 0.5), ("m_max", 4))
SHIFTED_LTYPES = (("S", 0.15), ("D", 0.15), ("W", 0.50), ("M", 0.20))
CAUSAL_MAF_BINS = ((0.01, 0.05), (0.05, 0.2), (0.2, 0.5))

# ---------------------------------------------------------------- locus mixture (DESIGN §2.4)
LOCUS_TYPES = {"S": 0.30, "D": 0.30, "W": 0.25, "M": 0.15}
S_EXTRA = 0.5
D_EXTRA = 0.3
W_SHARED_SUB = 0.5
# r² / d strata of D loci (DESIGN v4 §8.6): r² edges [0, .1, .3, .5, .8, .95); d edges [2, 4, 8, 16, inf)
R2_BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 0.95))
D_BINS = ((2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, float("inf")))

# ---------------------------------------------------------------- pairing types (DESIGN §2.2, §2.4)
N1_GRID = (20_000, 100_000, 400_000)
N2_GRID_P1 = (100, 300, 1_000, 5_000)
N2_GRID_P2 = (50_000,)     # DESIGN v4: P2 fixed at N1 = 100k, N2 = 50k
N1_GRID_P2 = (100_000,)
P3_EQTL_N = (300, 1_000)
P3_LAB_N = 100_000
P3_EQTL_SHARE = 0.5        # README D7: share of eQTL x eQTL among P3 loci
P3_BOTH_SIGNS = True       # training / validation: rho sign random (DESIGN v4 §2.4); test: positive
MAX_RP = 0.9               # assert per-row overlap correlation |r_p| <= 0.9 (R3-stat MF1)
FACTOR_MIN_N = 20_000      # N >= this -> factor form (DESIGN §2.2)

# ---------------------------------------------------------------- mismatch (DESIGN §2.3)
L4_FLIP = 0.005
L4_MISSING = 0.05

# ---------------------------------------------------------------- rho-hat (DESIGN §2.4)
# D6(d): truncated MLE on |z| < 2 SNPs after masking every SNP in LD (r² > 0.1, LD given to methods) with a
# SNP of max(|z1|,|z2|) > 4; SE by delete-one-block jackknife over RHO_JK_BLOCKS contiguous blocks (README D14)
RHO_MASK_R2 = 0.1
RHO_Z_SIGNAL = 4.0
RHO_JK_BLOCKS = 10
RHO_PRUNE_R2 = 0.1         # DESIGN §2.4 pruning kept (unpruned composite MLE: SD +50%, jackknife SE 1.6-2.7x too small)
RHO_RESID_MIN_NREF = 1000  # residualise strong signals only with accurate LD (L0 in-sample, or n_ref >= 1000)
RHO_ZMAX = 2.0
RHO_GW_NSNP = 20_000
RHO_GW_POLY = 0.01
RHO_GW_POLY_SD = 2.0       # lambda ~ N(0, 2^2) for the 1% polygenic SNPs, independent across traits

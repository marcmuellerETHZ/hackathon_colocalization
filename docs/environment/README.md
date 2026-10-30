# Software environment

All commands below run from the repository root. Install locations (`.venv/`, `env/renv/`, `env/rlib_*`,
`env/tools/`) are local and git-ignored; only `env/activate.sh` is tracked.

The project ran on one Linux node (64 CPU cores, 128 GB RAM, one NVIDIA RTX 4090, driver 580.178.04).

| Component | Version |
|---|---|
| Python | 3.12.3 (`.venv`, created with uv) |
| PyTorch | 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24 = 92400; Triton 3.8.0) |
| R | 4.5.3 (conda-forge, micromamba env `env/renv`) |
| plink2 / bcftools / htslib | 2.0.0a.7.8 / 1.24 / 1.24 (conda env `env/tools`) |

## Files
| File | What it is |
|---|---|
| `requirements.txt` (repository root) | the Python packages the code imports, pinned to the versions used |
| `python_requirements.full.lock` | full `pip freeze` of the project's `.venv` (90 packages) |
| `conda_R_env.lock.yml` | full conda export of the R environment (`r-base 4.5.3`, `r-coloc 5.2.3`, `r-susier 0.14.2`, `r-rfast`, `r-mashr`, ...) |
| `conda_tools.lock.yml` | conda export of the command-line tools env (plink2, bcftools, tabix) |
| `R_packages_pinned.tsv` | every R method package: version, source, commit, library |
| `BASELINES_INSTALL.md` | the project's install record for every baseline method, with call signatures and checks (copied verbatim; paths refer to the original project) |

## R packages used by the benchmark
| Package | Version | Source | Library |
|---|---|---|---|
| coloc | 5.2.3 | conda-forge `r-coloc` (CRAN) | main |
| susieR (standard coloc-SuSiE) | 0.14.2 | conda-forge `r-susier` (CRAN 0.14.2) | main |
| susieR (mismatch-aware, `R_finite` / `R_mismatch = "eb_mix"`) | 0.16.6 | GitHub stephenslab/susieR @ 97feba561dda5402fb9d70fabc715cb49030101f | separate library `env/rlib_susie016` |
| mvsusieR | 0.3.0 | GitHub stephenslab/mvsusieR @ 2b05b9a7ab4af2121218a74f08c33a59af2bbe53 (built against susieR 0.16.6) | `env/rlib_mvsusie` |
| colocboost | 1.0.10 | CRAN (= GitHub StatFunGen/colocboost @ 9f82c2a) | main |
| colocPropTest | 0.9.3 | CRAN (= GitHub chr1swallace/colocPropTest @ f8dfc43) | main |
| prop.coloc | 2.0.0 | GitHub ash-res/prop-coloc @ d7f0938f12e182a6b444ec9225e409b1a1cdc210 | main |
| hyprcoloc (tier 2, not benchmarked) | 0.0.2 | GitHub jrs95/hyprcoloc @ 0348bbd | `env/rlib_hyprcoloc` |
| SharePro_coloc (Python) | commit 059caab (v5.0.0) | GitHub zhwm/SharePro_coloc, cloned to `env/src/SharePro_coloc` | — |

**Two susieR versions.** Both are needed: 0.14.2 for standard coloc-SuSiE (coloc 5.2.3 was built against 0.14.x)
and 0.16.6 for the mismatch-aware model. They must never be loaded in one R session. The project installed 0.16.6
into its own library and selects it with `R_LIBS` (the main library still supplies coloc 5.2.3):

```bash
# main R environment (R 4.5.3, coloc 5.2.3, susieR 0.14.2, Rfast, mashr, cpp11armadillo, ...)
micromamba create -p env/renv -f docs/environment/conda_R_env.lock.yml
# susieR 0.16.6 in a separate library
git clone https://github.com/stephenslab/susieR env/src/susieR && git -C env/src/susieR checkout 97feba561dda5402fb9d70fabc715cb49030101f
mkdir -p env/rlib_susie016 && env/renv/bin/R CMD INSTALL --library=env/rlib_susie016 env/src/susieR
# use it
R_LIBS=env/rlib_susie016 R_PROFILE_USER=/dev/null env/renv/bin/Rscript -e 'packageVersion("susieR"); packageVersion("coloc")'
```

The Python R-worker (`src/baselines/rworker.py`) finds these through the defaults above or through
`COLOC_BENCH_RSCRIPT`, `COLOC_BENCH_RLIB_SUSIE016` and `COLOC_BENCH_RLIB_MVSUSIE`.

## Python
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# exact replica of the project environment:
uv pip install --python .venv/bin/python -r docs/environment/python_requirements.full.lock
```
`source env/activate.sh` puts `.venv`, `env/renv` and `env/tools` on PATH and keeps caches inside the repository.

To recompute the benchmark numbers (`scripts/recompute_benchmark.py`) only numpy, scipy, pandas and pyarrow are
needed. The recommended pipeline (`mvp/run_coloc_mm.R`) needs only R with coloc 5.2.3 and susieR 0.16.6.

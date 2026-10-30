# Datasets

No raw data are in this repository. This folder holds the small files that define the benchmark's data: the
LD-block and gene splits, the frozen test-set and methods configurations, and the checksums of every downloaded
and generated input. The scripts listed below download and rebuild everything else.

## `splits/`: train / validation / test splits (DESIGN.md §3)
| File | What it is | Built by |
|---|---|---|
| `blocks_v1.tsv` | LD blocks (tiled from MacDonald et al. 2022, GRCh38) with their split label: train / valid / test (60/20/20 per chromosome) | `python -m data.make_splits` |
| `genes_v1.tsv.gz` | GENCODE v39 genes with their gene windows (TSS ± 500 kb, clipped to the same-split block run) and split | `python -m data.make_splits` |
| `pools_v1.tsv` | 1000 Genomes sample IDs with their pool (A, B, FIN, ASW_ACB, L6c), from which the simulator builds target populations and LD reference panels (DESIGN.md §2) | `python -m data.pools` |
| `SHA256SUMS` | sha256 of the three files | `python -m data.make_splits` |

## `configs/`: frozen configurations
| File | What it is |
|---|---|
| `testset_v1.yaml` | test-set definition v1 (DESIGN.md v4 §8.6): cells, per-cell sizes, seed, endpoints, slow-method subsamples. The test run used it once |
| `testset_v2.yaml` | test-set definition v2 (docs/DESIGN_P2.md §6), with its own lock. It was drafted and frozen, but never run |
| `test_methods_v1.yaml` | the frozen methods file of the test run: methods, timeouts, operating points |
| `test_methods_v1_d17cut1.yaml` | the same without mvSuSiE, used after decision D17/D21 |
| `proposals/` | drafts and dry-run methods files that led to the frozen ones |
| `FROZEN_SHA256`, `FROZEN_P2_SHA256` | sha256 hash lists of the hash-protected definitions (design, metrics, test-set generator, test-run pipeline, the configs above) for phase 1 and phase 2 |
| `real_test_ids_v1.sha256`, `real_test_meta_v1.sha256` | sha256 of the frozen real-data test ID lists and their metadata. The lists themselves are not shipped: they are built from FinnGen and eQTL Catalogue data |

The hash lists name paths in the original project layout (`configs/...`, `splits/...`, `DESIGN_P2.md`).
`python scripts/check_frozen.py` resolves them to `datasets/` and `docs/`. In this release it reports two
entries as `CHANGED` (`src/eval/testrun.py`, `scripts/eval_test.py`): both files were edited in phase 2 after the
v1 test run (the test-set v2 lock, docs/DESIGN_P2.md §6). The source project's final commit has the same two
differences. Every other entry matches. The frozen test-run entry points (`scripts/eval_test.py`,
`src/sim/testset.py`, `src/eval/testrun.py`, `src/data/real/extract.py`, `scripts/run_benchmark.py`) are kept
byte-identical to the project's final state. Their defaults therefore still say `configs/...`, so give paths
under `datasets/configs/` explicitly. Test set v1 has been used once and must not be run again.

## `checksums/`: what was downloaded and generated
| File | What it is |
|---|---|
| `downloads_sha256.tsv` | path, size and sha256 of every eQTL Catalogue and FinnGen file the real-data arms downloaded (`scripts/real/run_downloads.sh`); it mirrors the per-directory SHA256SUMS below. DATA.md refers to it by its original path, `scripts/real/downloads_sha256.tsv` |
| `data/raw/1kg30x/SHA256SUMS` | 1000 Genomes 30x phased GRCh38 VCFs (`scripts/download_1kg30x.sh`; MD5s are checked against IGSR by `scripts/verify_1kg_md5.sh`) |
| `data/raw/annot/SHA256SUMS` | LD blocks (MacDonald et al. 2022) and GENCODE v39 (`scripts/download_annotations.sh`) |
| `data/raw/genmap/SHA256SUMS` | Beagle GRCh38 genetic map (`scripts/download_genetic_map.sh`) |
| `data/raw/eqtlcat/**/SHA256SUMS` | eQTL Catalogue r7 metadata and the nine QTD datasets used (`scripts/real/download_eqtlcat.sh`) |
| `data/raw/finngen/r12/**/SHA256SUMS` | FinnGen R12 summary statistics, SuSiE fine-mapping, LD-matrix indices, documentation (`scripts/real/download_finngen.sh`) |
| `data/processed/sets/SHA256SUMS` | the simulated locus sets (HDF5) and truth tables of the screen, validation and L0* sets (`src/sim/`) |

The paths mirror the original project's `data/` tree. A re-download is placed there, so `sha256sum -c` can be
run in each directory.

## Data sources and terms
| Data | Terms | Here |
|---|---|---|
| 1000 Genomes 30x phased GRCh38 (IGSR; Byrska-Bishop et al. 2022) | open, no restrictions | not redistributed; derived simulation outputs are in `results/`; sample IDs in `splits/pools_v1.tsv` |
| eQTL Catalogue r7 (Kerimov et al. 2021) | CC BY 4.0 | not redistributed; method outputs and identifiers are in `results/` |
| FinnGen R12 public summary statistics (Kurki et al. 2023) | free use after the access form; **no redistribution**; acknowledge and cite | not redistributed; only method outputs and identifiers, with raw statistics stripped (`results/README.md`) |
| LD blocks, MacDonald et al. 2022 | **no licence asserted** | not vendored; downloaded by `scripts/download_annotations.sh`; `splits/blocks_v1.tsv` holds the project's own tiled blocks |
| GENCODE v39; Beagle GRCh38 genetic map | open / free for use | not vendored; downloaded by script |

## Not included
| Original project folder | Size | What it is | Where to get it |
|---|---|---|---|
| `data/` | 911 GB | raw 1000 Genomes, eQTL Catalogue and FinnGen downloads, and the processed locus sets built from them. FinnGen data may not be redistributed | download and build scripts in `../scripts/` (below) |
| `env/`, `.venv/` | 12 GB | the installed R, Python and CUDA environments | `../docs/environment/` |
| `runs/` | 13 GB | the job queue's per-job code snapshots, duplicates of `src/` | not needed |
| `.worktrees/` | | the agents' temporary per-task git checkouts | not needed; their merged work is in this repository |

## Rebuilding everything
Run from the repository root, after `source env/activate.sh` (environment: `docs/environment/README.md`). The
1000 Genomes panel alone is 28 GB.
1. Reference data: `scripts/download_1kg30x.sh`, `scripts/verify_1kg_md5.sh`, `scripts/prep_1kg_pgen.sh`
   (VCF to pgen), `scripts/download_annotations.sh`, `scripts/download_genetic_map.sh`.
2. Pools, variants and splits: `PYTHONPATH=src python -m data.pools`, `python -m data.variants`,
   `python -m data.make_splits` (`src/data/README.md`). The rebuilt splits must match `splits/SHA256SUMS`.
3. Simulated locus sets: `src/sim/` (`src/sim/README.md`; `python -m sim.testset --config
   datasets/configs/testset_v1.yaml ...` for the evaluation grids).
4. Real-data arms: `scripts/real/run_downloads.sh` (eQTL Catalogue, FinnGen; FinnGen needs its access form
   accepted first), then `scripts/real/build_real_arms.sh` (`src/data/real/README.md`).

(Human-written text starts)
This is an exercise in assessing the bioinformatics capabilities of agentic AI. It is part of a hackathon run by the Beltrao lab at ETH Zurich, where we used Claude to run bioinformatics projects (https://github.com/evocellnet/ai-hackathon/). All results in this project were generated entirely by AI with minimal human intervention. This is not intended to be viewed as a traditional publication and bioinformatics tool.
The goal of this project was to come up with a new analysis tool to infer colocalization of genomic variants identified by GWAS and QTL studies. It was executed by Claude Opus 5.5 with delegation to Opus 5.5/Sonnet 5.5 subagents, and finished in 50h of wall time on a computational cluster with 64 CPUs with 128 GB RAM and one RTX4090 GPU.
Currently none of the results have been verified by a human. We plan to verify the agent's methodology and benchmarking, and we will update this section accordingly.
The rest of this repository is entirely AI generated. It includes code for implementation of the predictive model and an AI-written report in the style of a journal article, detailing the final method, the path followed to reach it, and benchmarking against other methods.
(Human-written text ends)

# coloc-agentic-benchmark

A benchmark of GWAS–molecular-QTL colocalization methods under LD-reference mismatch and sample overlap, and the
new methods an AI agent team built and tested against it. The agents pre-registered the design, split the genome
into LD-block train / validation / test sets, and ran the test set once. This repository holds the full research
record, the code, the predictions behind every reported number, a minimal usable pipeline and a manuscript.

## Key result
No new method beat the best existing one. The best method under realistic LD mismatch was **mismatch-aware
coloc-SuSiE**: susieR 0.16.6 with `R_finite = n_ref` and `R_mismatch = "eb_mix"`, followed by coloc's
`coloc.bf_bf`. This is an existing method.

Test set (13,207 held-out simulated loci, single run), realistic setting (27 mismatched-LD cells), AUPRC for a
shared causal variant vs two distinct well-powered variants:

| Method | AUPRC S-vs-D, realistic [95% CI] | AUPRC S-vs-D, in-sample LD |
|---|---|---|
| mismatch-aware coloc-SuSiE (susieR 0.16.6) | **0.900** [0.884, 0.914] | 0.962 |
| coloc.abf | 0.860 [0.846, 0.874] | 0.856 |
| coloc-SuSiE (susieR 0.14.2) | 0.717 [0.697, 0.741] | 0.966 |

- LD-reference mismatch, not sample overlap, is what breaks fine-mapping-based colocalization. The mismatch-aware
  model recovers most of the loss.
- Its calling thresholds were validated in phase 2: t_S = 0.840, t_D = 0.732 (recall S 0.78, recall D 0.80; 2.5%
  of weak loci and 9.7% of distinct loci called shared). It is slow: about 16 CPU-seconds per locus.
- The project's four new method families (JointSuSiE; ExactColoc and its mismatch-aware version; a GPU
  coloc-SuSiE; NeuralColoc) failed their pre-registered tests and are reported descriptively only. REPORT.md has
  the full account, including what went wrong.

## Where to start
- **To use the method:** `mvp/`. It holds the recommended pipeline (`mvp/run_coloc_mm.R`) with an example, its
  tests, and `mvp/recompute_benchmark.py`, which recomputes every simulated test-set number from `results/` with 0
  mismatches.
- **For the paper:** `manuscript/`.
- **For the research record:** `REPORT.md` (the agents' final report) and `RESEARCH_TREE.md` (the PI agent's
  decision log D0–D29), with `DESIGN.md`, `DATA.md`, `LESSONS.md` and `LITERATURE.md`.
- **For the process:** `docs/` (prompts, decisions, phase-2 design, agent rules, per-task reports, orchestration,
  environment) and `reviews/` (design reviews, independent verifications, retrospectives).

```bash
pip install -r requirements.txt                   # or only numpy, scipy, pandas, pyarrow, pytest
python mvp/recompute_benchmark.py                 # REPRODUCED: 28363 values ...; mismatches 0
python -m pytest                                  # tests/smoke + mvp/tests; pipeline tests need R (mvp/README.md)
```

## Layout
| Path | Content |
|---|---|
| `README.md`, `PROMPT.md` | this file; the original project brief, verbatim |
| `REPORT.md`, `RESEARCH_TREE.md` | the agents' final report (phases 1 and 2); the research tree and decision log |
| `DESIGN.md`, `DATA.md`, `LESSONS.md`, `LITERATURE.md` | pre-registered design v4; every dataset with source, checksum and licence; process lessons; literature review |
| `mvp/` | the minimal usable deliverable: recommended pipeline, example, recompute wrapper, tests, README |
| `manuscript/` | the manuscript (PLOS Computational Biology format), S1/S2 Text, figures |
| `results/` | test-run predictions and bundle (`test_v1/`), truth tables (`truth/`), phase-2 validation (`phase2_validation/`), NeuralColoc models (`models/famC/`), recompute output (`recompute/`); all other screening, validation, kill-test, method-family, dry-run and verification outputs (`results/README.md`) |
| `datasets/` | LD-block and gene splits, frozen configs and hash lists, checksums of all downloads; no raw data |
| `src/` | simulator, evaluation, baselines, method families A/B/C/G, data preparation |
| `scripts/` | benchmark, test-run, verification, download and prep scripts; `real/` arm builders; `famB_eval/`, `famC_eval/`; `orch/` (the agents' job queue); `release_build/` |
| `tests/` | `smoke/` release smoke tests; the project's original test suite, one folder per module |
| `docs/` | decisions, phase-2 design, agent rules, interfaces, prompts, literature notes, per-task reports, orchestration, environment; the agents' job/task/artifact registries and logs (`agent_logs/`) |
| `reviews/` | design reviews, independent verifications (with the verifiers' scripts) and retrospectives |
| `requirements.txt`, `env/activate.sh` | pinned Python dependencies; environment set-up script (full environment: `docs/environment/`) |
| `CITATION.cff`, `LICENSE`, `NOTICE` | citation; MIT licence for code; CC BY 4.0 for results and text, with third-party terms |

All paths resolve relative to the repository root. Shell scripts accept `COLOC_BENCH_ROOT`; the R worker takes
`COLOC_BENCH_RSCRIPT`, `COLOC_BENCH_RLIB_SUSIE016`, `COLOC_BENCH_RLIB_MVSUSIE` and `COLOC_BENCH_TMP`. The record
files keep the original project's paths (`configs/`, `splits/`, `results/eval_test/...`); `datasets/README.md`
and `results/README.md` map them. Re-running the whole project needs the downloads and the full environment
(`datasets/README.md`, `docs/environment/README.md`). The test lock in `src/eval/io.py` blocks reading test-split
loci outside an authorised test run; test set v1 has been used once and must not be used for model selection.

## Data sources and licences
| Data | Use | Terms | In this repo |
|---|---|---|---|
| 1000 Genomes 30x phased GRCh38 (IGSR; Byrska-Bishop 2022) | simulation backbone, LD references | open, no restrictions | derived only: simulated predictions and truth, example LD factor (150 reference samples) |
| eQTL Catalogue r7 (Kerimov 2021) | real arms R2, R3b, R4b | CC BY 4.0 | method outputs and identifiers only |
| FinnGen R12 public summary statistics (Kurki 2023) | real arm R3b | free use after the access form; no redistribution; acknowledge and cite | method outputs and identifiers only; raw statistics removed (`results/README.md`) |
| GEUVADIS (via the eQTL Catalogue and 1000G) | real arm R4b | CC BY 4.0 / EBI terms | method outputs only |
| LD blocks, MacDonald et al. 2022 | split units | no licence asserted | not vendored; downloaded by `scripts/download_annotations.sh` |
| GENCODE v39, Beagle GRCh38 genetic map | gene windows, recombination | open / free for use | gene windows derived; map not vendored |

- **Code:** MIT (`LICENSE`).
- **Results, datasets files, the research record and the manuscript:** CC BY 4.0, subject to the third-party terms
  above (`NOTICE`).
- **Not included:** raw data, the full simulated locus HDF5 sets, model training shards and anything restricted.
  The scripts regenerate them (`datasets/README.md`).

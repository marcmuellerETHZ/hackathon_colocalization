# INTERFACES — shared contracts between modules (owner: PI; change only via a decision note)

All code is Python 3.12 in `src/` (package root `src/`, run with `PYTHONPATH=src`), or R scripts called from Python wrappers. Source `env/activate.sh` first. Never write outside the project directory, except temporary files under `/tmp/claude-*`.

## 1. Directory ownership (one owner per directory)

| Path | Owner | Content |
|---|---|---|
| `src/data/` | data agent | variant QC, pools, blocks/splits, windows, real-data harmonisation |
| `src/sim/` | simulation agent | mosaic population, LD worlds, generator, oracle classes, cell builder |
| `src/stats/` | simulation agent | ρ̂ estimator, power functions |
| `src/eval/` | eval agent | metrics, adapters, run harness, manifests, ledger, test lock |
| `src/baselines/` | baseline agent | wrappers for every baseline (R and Python) |
| `src/famA/`, `src/famG/` | Family A/G agent | GPU SuSiE-RSS (G) and JointSuSiE (A) |
| `src/famB/` | Family B agent | ExactColoc |
| `src/famC/` | Family C agent | NeuralColoc |
| `tests/<module>/` | same as module | pytest tests |
| `configs/` | PI (agents propose) | YAML configs; `configs/testset_v1.yaml` is frozen |
| `results/` | run harness | `results/<run_id>/` (manifest.json, predictions.parquet, metrics.json), `results/LEDGER.tsv` |

## 2. Locus input format (what every method receives)

A **locus set** is a single HDF5 file (`h5py`), one group per locus `/<locus_id>/`. Never write one file per locus. Datasets per locus:

| Name | Type | Meaning |
|---|---|---|
| `variant_id` | str[p] | `chr:pos:ref:alt` (hg38) |
| `pos` | int64[p] | |
| `z1`, `z2` | float64[p] | z-scores of trait 1 (GWAS-like) and trait 2 (QTL-like); sign relative to alt |
| `beta1`, `se1`, `beta2`, `se2` | float64[p] | derived as in DESIGN §2.2 (sdY = 1) |
| `n1`, `n2` | float64[p] | per-SNP sample size (constant unless L5) |
| `maf1`, `maf2` | float64[p] | in-sample MAF of each study |
| `maf_ref` | float64[p] | reference-panel MAF |
| `ld_type` | attr | `"dense"` or `"factor"` |
| `R` | float32[p,p] | dense LD (if dense) |
| `X_ref` | float32[n_ref,p] | standardized reference dosages (if factor); R = XᵀX/n_ref |
| `n_ref` | attr int | number of reference individuals |

Locus attributes: `rho_hat`, `rho_se`, `split`, `block_ids`, `mismatch_level`, `pairing_type`, `cell_id`, `window_chr`, `window_start`, `window_end`.

**Truth** is stored in a *separate* file (`<locusset>.truth.parquet`) that methods never open. It holds: causal indices and effects per trait, μ, oracle class, d, r², class-determining (μ1, μ2), true ρ. A test asserts that no method module opens truth files.

## 3. Method output format

Each method run writes one parquet file (`predictions.parquet`) with one row per expected locus:

| Column | Meaning |
|---|---|
| `locus_id` | |
| `method` | |
| `s` | ungated S-vs-D ranking score |
| `P_S`, `P_D`, `P_W` | probabilities; NaN if undefined |
| `E1`, `E2` | exclusion probabilities, NaN if not available |
| `call` | `S` / `D` / `insufficient` at the method's thresholds; filled by the harness after thresholds are fixed |
| `status` | `ok` / `nonconverged` / `failed` / `timeout` |
| `runtime_s` | wall-clock for this locus, including LD loading |
| `extra` | JSON string with method-specific details (e.g. PP0–PP4, number of CSs) |

## 4. Python method API (used by the harness)

```python
class Method:
    name: str
    def fit_batch(self, loci: list[LocusData]) -> list[dict]:  # dict keys = output columns above
        ...
```

`LocusData` is a dataclass mirroring §2, loaded by `src/eval/io.py` (owner: eval agent). GPU methods receive large batches; CPU/R baselines are parallelised by the harness over processes.

## 5. Seeds

SEEDS = [20261026, 20261027, 20261028, 20261029, 20261030], with uses as in DESIGN §9. Any random draw takes an explicit `numpy.random.Generator` or `torch.Generator` seeded from these via `src/common/seeds.py`.

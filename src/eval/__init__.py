"""Evaluation infrastructure (owner: eval agent). See src/eval/README.md.

Modules:
  io        -- LocusData + HDF5 locus-set reader/writer (INTERFACES §2) and the test-split guard.
  truth     -- truth parquet reader/writer. Method code (src/fam*, src/baselines) must NEVER import it.
  metrics   -- the single metrics implementation (DESIGN §8).
  adapters  -- generic score adapters (DESIGN §4.3); per-method adapters live with the wrappers.
  harness   -- run a Method over a locus set; manifests, ledger, machine lock.
  reference_methods -- trivial reference Method (lead-SNP rule) used to exercise the harness.
"""

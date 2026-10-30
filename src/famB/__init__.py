"""Family B -- ExactColoc: exact Bayesian enumeration (<= 2 causal variants per trait) on GPU.

DESIGN v3 Sec. 5 B. Modules:
  constants  power thresholds (DESIGN Sec. 1)
  enum       configuration tables, support-space Bayes factor kernel, two-pass streaming sums + sampler
  events     Gaussian effect posteriors of sampled configurations, oracle-class / exclusion events
  screen     per-trait-union screen with lead-removed top SNPs and strength-aware duplicate collapse
  priors     coloc / Gaussian-EB / simulator-prior (scale mixture) priors
  method     ExactColoc Method (INTERFACES Sec. 4)
"""

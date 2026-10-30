"""Frozen power thresholds (DESIGN Sec. 1) on the z (noncentrality) scale."""

TAU1 = 6.29  # powered, trait 1 (alpha = 5e-8, power 0.8)
TAU2 = 5.26  # powered, trait 2 (alpha = 1e-5, power 0.8)
WEAK1 = 4.61  # power < 0.2 below this, trait 1
WEAK2 = 3.58  # power < 0.2 below this, trait 2
D_MIN = 2.0  # identifiability index below which a D-type pair is U

CAT_NAMES = ("H0", "H1", "H2", "H3", "H4", "H34")
# H0 no signal; H1 trait 1 only; H2 trait 2 only; H3 both, no shared variant;
# H4 both, all causal variants shared (S1 == S2); H34 >= 1 shared variant plus >= 1 trait-specific variant.
CLASS_NAMES = ("S", "U", "D", "W", "G")

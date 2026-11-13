# Frozen old-target identity on the new validation photometry

Produce the explicitly named `response12_old_target` control for every frozen
field/common validation object before opening its reference outcomes. Use the
original response12 native SED and shared calibration12, the authenticated
v3.0 refit12_s4096 population posterior and its native-likelihood importance
correction. Retain all original shared population draws, fitted within-class
SED tilts and binary branches. Do not replace this control by the new fixed-kernel
zero-correction target fit; those are different statistical identities.

Assign original field priors by the original ordered footprint geometry, as in
the already registered seen-development target-prior diagnostic. Outside those
fields use disjoint10000+sky-cell IDs and sample the original .35 field effect
from each retained global hypermean with seed[316101,12,field_id]. The new
reference photometry updates each star's conditional distance/class posterior;
it does not refit or reweight the population posterior. This is the same
explicit conditional-transfer/cut identity as the earlier target product,
not a joint population refit on the new validation cohort. Known field and new
field ownership, historical target overlap and all shared indices are exported.

Use no new distance/class/membership outcomes. Native TYPE=PSF is mandatory.
Start from the registered reference-refinement resolution for each identity:
compare global[-20,50] grids with twice that step and that step, including exact
distance-bin boundaries. If necessary use bounded.005/.0025/.00125 refinements,
retaining every attempt. Require joint-probability change<.003, mu quantile
change<.03mag, exact joint normalization<1e-8 and interleaved shared-draw
marginal-probability change<.005. A sampling failure is not repaired by claiming
finer distance quadrature. Numerical flags never delete an object or select a
new model. Final paired scores wait for the full candidate/method freeze.

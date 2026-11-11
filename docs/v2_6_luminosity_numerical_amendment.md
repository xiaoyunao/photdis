# Numerical-only luminosity integration refinement

Initial production 61129363 completed successfully. Its SPDist source-offset
95% interval collapsed to a single .05-mag grid point. Preserve all initial
products in data/v2_6; this is a numerical resolution problem, not evidence for
zero source-scale uncertainty. Same observations, source/role/PSF cuts, parameters,
uniform prior bounds and 4096 predictive draws. No new scientific candidate.

Use 161 offset points on [-1,1] for BOTH source and poor offsets (step .0125 mag)
and 25 scatter points on [0,.6] (step .025 mag). The joint grid has 648025 points.
Compute the same signed likelihood, and also compare 32/64-node quadrature at the
posterior-mean parameters. Write all refined products into a separate directory
data/v2_6/luminosity_refined. Initial summaries remain archived in delivery.
Final reports and validation explicitly use refined outputs and show numerical
comparison; do not mix initial/posterior predictive versions.

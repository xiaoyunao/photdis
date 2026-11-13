# Matched shared-target-prior diagnostic

Registered before target-prior diagnostic outcomes. Use exactly the frozen
pilot32 and diagnostic1542 identities; nominal/reference and response12/reference
controls already exist. Add response12/target only, with identical native flux,
mask, calibration draw12 and fixed SED bytes. No truth enters the new inference.

Use all interleaved samples and exact native likelihood correction weights from
v3.0 refit12_s4096. Each draw's object posterior is normalized by that object's
own evidence before averaging. Retain all draw probabilities and full log PDFs;
no average-theta plug-in, weight flattening or response-mixture convergence claim.
This is conditional on the response12 fit, whose finite response-library ESS1
remains a limitation.

Assign original target-field effects only by actual original field footprints,
with original first-field overlap ownership. Reference cell IDs are not field IDs.
Outside those footprints use a disjoint10000+cell field namespace and a shared
new-cell Gaussian field effect drawn from the frozen hypermean, sigma.35,
seedSequence[316101,12,field]. Export geometry/extrapolation and exact overlap with
the26000 target observations. New cells are not separately fitted from references.

Native integration uses the response12 reference adaptive grid plus[-20,50] at
.05mag and exact distance-bin boundaries. Repeat at all interval midpoints;
require class×distance probability change<.003, quantile change<.03mag,
normalization/joint reconstruction error<1e-8 and weighted interleaved-half
probability delta<.005. Keep coarse and fine sidecars and numerical flags.

Evaluate the same source-error-convolved references and signed Gaia, then pair
with the same identity's reference-prior control using original design weights
and sky blocks. These are seen development diagnostics. They do not constitute
new final validation, a refit on all catalog stars or calibrated physical counts.

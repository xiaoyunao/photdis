# v3.1 independently regenerated response library

Register a second32-draw library before target evidence is evaluated. This is
independent Monte Carlo regeneration **conditional on the same training sources**,
not32independent datasets or an independent absolute-scale calibration.

Re-bootstrap the original SpecDis and SPDist source-consistency calibration
using their exact saved input identities, native PSF qualification, signed Gaia
likelihood and original fitting functions. Use one common set of sky weights
for both source fits, seed319103. Keep the original point calibration fixed.
Bootstrap SED source identities and sky weights independently with seed319101;
regenerate band zero points/field dust from the original declared priors with
seed319102. New global draw IDs1000..1031 ensure new-cell dust draws also differ
from the old library. Do not perturb each SED component independently.

For every draw, refit the same128component SED from the same original nominal
model using the same joint distance quadrature and weighted EM implementation.
Retain iteration checkpoints and the original convergence rule (at least4
iterations; objective change<1e-4*(1+absolute objective)); maximum64iterations.
All32draws are fixed in advance. No response may be removed because it predicts
the target poorly. Numerical failures stay in the denominator and must be
resolved or explicitly prevent normalized-library claims.

Compare against all32old converged responses on identical preselected target
and seen-reference probe identities, with identical likelihood/prior/selection
definitions. Decompose log scores by class availability, magnitude, colors,
source and sky; compare inferred distance/count sensitivity and response ESS.
Conditional predictive evidence at a fixed prior is explicitly different from
population-marginal evidence after refitting the target hierarchy. Neither may
be silently substituted for the old32-response target evidence. If an importance
transfer lacks support, retain that failure and fit the required conditional
population instead of flattening weights or dropping responses.

This experiment tests finite-library sensitivity and missing systematic support.
An independently regenerated ESS near1 still does not establish that the true
response lies inside the assumed reference/SED/systematic model. Do not combine
libraries with ad hoc equalization, temper likelihoods, or claim that more draws
repair an inadequate scientific model.

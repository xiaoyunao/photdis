# Matched-response evidence and predictive summary

Retain all64 registered response states and all384 frozen probe identities.
Incomplete states produce an inventory only. Within-original32, within-new32
and pooled64 summaries each require every target state in that named set to
pass the registered numerical checks; a failed draw cannot be removed and the
remaining posterior renormalized. Use actual conditional128-object flux log
evidence and uniform Monte Carlo prior over the named set. No weight flattening.

For each set propagate native coarse/fine, actual second-seed entry and actual
previous-level entry through the full summed log evidence. Report maximum
response-weight change, L1 change, ESS and normalized evidence differences;
require maximum weight change<.01 and log-evidence change<.05nat for a numerically
accepted response-weight summary. Failure leaves all raw results visible.
This is a numerical check on already registered scientific priors, not a new
selection or response model.

Decompose per-draw target log evidence by observed magnitude, color, band
availability and sky, using the previously registered diagnostic bin edges.
The disjoint groups within each axis must sum to the actual total evidence.
Decompose seen-development predictions by the same observable axes plus
reference class and metallicity. Retain source-separated scores and signed
Gaia scores/PIT. Pair every response against original12 on identical objects,
using combined inclusion probabilities and sky-block uncertainty. These are
seen-development associations, not independently measured calibration accuracy.
One shared response state applies to the entire catalog. Mixture source/Gaia
log scores are log-sums of conditional scores, not averaged log scores; class
and distance probabilities are linear mixtures. Mixture quantiles cannot be
inferred from averaging per-response quantiles and will not be reported here.

Compare class-by-distance counts on the same128target identities across the
original/new/pooled sets, with one global response state shared by allstars.
Compute exact categorical conditional covariance plus response covariance and
exact response-mixture Poisson-binomial marginals for each count component.
Marginal count intervals are not independent; deliver the full covariance.
Check actual numerical variants for per-object joint-probability change<.003
and mean-count change<.5. These are conditional128-object probe counts, never
an expanded target population or a new26kpopulation fit. Retain finite-response
ESS and half-library discrepancies as support diagnostics, not MC convergence
of an imagined continuous response model.

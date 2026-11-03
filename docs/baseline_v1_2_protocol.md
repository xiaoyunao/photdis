# Baseline v1.2: color separability and distance decisions

Registered before new fitting/scoring; source: [full Pro review](pro_review_dbeac65.md).
Preserve v0/v1/v1.1 and all full catalogs. Only existing references, the fixed
26,000 optical-parent sample and fixed7,672 Gaia sample are processed on allocated
NERSC CPU resources. No clusters, new priors, additional bands or parameter search.

## Frozen scope and two new classifiers

Original spatial train/cal/test remain unchanged; test is development validation.
Six configurations A,B,C,A_on_B,A_on_C,B_on_C preserve same-training comparisons.
Use original fit-eligible training objects with reliable labels0/1/2 (dwarf,
evolved,transition). Unknown labels remain in evaluation and distance-risk counts.
Both new classifiers use exactly the same identities and weights: class k receives
N/(3*N_k) per object; thus each class has equal total weight. No cal/test fitting.
1. Weighted standardized colors, degree2 features, logistic C=1,max_iter=2000.
2. Fixed random forest256 trees,min_leaf32,max_depth20,max_features1.0,seed20261103.
These are ranking scores, not population-calibrated probabilities. The existing
natural, post-hoc-balanced and luminosity-evidence gates remain unchanged.

## Predictions and working points

Frozen luminosity experts are evaluated with identical identity-keyed Sobol128
band noise, including shared-g shifts. Cache all four conditional PDFs. New gate
weights condition on nominal observed colors (as v1.1 evidence gate); support
failure transfers mass to the fixed broad expert. Existing natural/balanced
mixtures retain their per-draw weights and original missing-label fallback.
Raw natural/balanced scores average the existing per-draw class scores before
missing-label/support fallback; report both raw scores and final four weights.
The fourth component is a broad distribution, not a fourth stellar population.

For evolved and transition separately, cal sets score>=threshold working points
with dwarf false-positive budgets1%,2%,5%. Choose the lowest threshold satisfying
the empirical budget, handle ties conservatively, and never tune on test. Count
all rejected/selected populations and unknowns. These are comparison budgets,
not claims of guaranteed field purity. Report ROC/PR, counts, FPR/recall and
reference precision (known classes only), color/r/error subgroups, and uncertainty
from finite counts. Pair A_on_B/B and A_on_C/B_on_C/C on identical rows.

Compare mixture median to the conditional PDF/median of the photometrically
predicted maximum final-weight component (including broad when dominant). Never
use true classes to choose predictions. Report entire selected groups, including
misclassified and unknown sources, separately for Gaia/SPECDIS/SPDist/chosen.
Preserve complete PDFs and branch solutions; a rejected single-distance decision
means retain multiple solutions, not discard the object.

A fixed confidence x top-two-margin grid (config) defines single-distance groups,
requiring broad and selected-component support. Cal may recommend the largest
retained group only if chosen-reference severe-error fraction<=5%, each of three
known classes has>=30 references, and each class tail<=same-star wide+2pp.
Otherwise no all-class single-distance tier is adopted. Evaluate every fixed grid
point as a diagnostic; test never selects the point. Report mixture vs main-branch
risk, all true-class contamination and parent inclusion-weighted retained fraction
with sampling SE. Parent coverage weights do not correct reference selection.

## Complete observed-parallax check

Reuse fixed Gaia sample and inverse inclusion weights. Convolve empirical PDFs
with Gaia measurement errors. In addition to old observed-SNR-conditional PIT,
report unconditional PIT/log density and model probability assigned to the actual
SNR bin; report predicted versus observed weighted fractions for five SNR bins.
No positive-parallax cut. The eligible parent is quality/zero-point selected Gaia,
not the full optical parent; r>=22 has no validation. Include existing wide and
evidence PDFs plus fixed new-classifier mixture and main-component PDFs for A/B/C.
No shared-systematic p-value claims; aggregate and r/WISE/color subgroup diagnostics.
Verify density factorization and bin-mass sum,8/16-point quadrature, negative rows,
identities and weights. Complete scores do not prove independent absolute accuracy.

## Endpoint and artifacts

Answer fixed-budget recall/contamination, training-weight/nonlinearity contrast,
mixture-summary versus branch distance risk/coverage, and full Gaia consistency.
Deliver compact machine-readable results, figures including distance density/CMD,
model/source hashes and actual validation. Preserve v1 deployment; no full catalog
application in this bounded experiment, regardless of promising development results.

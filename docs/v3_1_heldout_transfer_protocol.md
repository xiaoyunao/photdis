# Fixed heldout-sky population transfer

The three whole fields12/4/6 were fixed by identity-independent SHA ranking before
target fits. Evaluate their unchanged application photometry under broad26000,
candidate3540 and heldout-training-cohort populations. Only the latter excludes
these objects and fields from population fitting. Broad/candidate predictions
are explicitly in-sample comparisons; no independent-validation claim follows.
All newly used objects require the frozen nativePSF role and identity checks.

Retain all34correction outcomes and the four named zero/point/evidence-bootstrap/
equal-bootstrap-cut views. Missing or failed states are not renormalized away.
Heldout fitted models draw newfield logits as global_logits+0.35normal, using
the same [320401,correction,field] seed convention as external predictions; one
global index is shared across all stars and all three heldout fields. Existing
fitted fields use their actual local posterior draws. No newfit uses heldout
photometry or predictive residuals.

Report both the entire heldout broad-SNR5 sample and its original candidate-cut
subset. Correct each observed-context likelihood by its matching broad/candidate
entry integral. The population-conditional selected density is R(theta)/A(theta).
Pointwise predictive scores average that ratio over the frozen shared posterior;
they are not ratios of independently averaged numerator and denominator.
Joint catalog prediction integrates the product with ONE shared global draw,
not the product of separately marginalized per-star predictions. Report actual
unit-object joint scores separately from generalized design-weighted joint
scores; the latter are not normalized physical catalog densities. Design weights
are normalized to the evaluated sample size, as in the registered fits.

Record importance support/ESS and interleaved-half joint predictive differences;
joint predictive support requires ESS>=100 and half logdensity change<0.2.
Low support is a limitation, never silently flattened weights. This diagnostic
does not use posterior predictive reweighting to replace the frozen cut-transfer
distance/class predictions. Per-object class x true-distance probabilities and
shared categorical/count covariance use those individually normalized cut draws.

Propagate the actual entry-seed/previous-level and registered native-fine
alternatives through training population weights, calibration-state weights,
heldout selected density, per-object probabilities and counts. Numerical gates:
per-object joint probability0.003, mean count0.5, pointwise logscore0.02; retain
all failures and joint predictive support flags. Fine-grid denominator is the
actual audited heldout subset. No heldout true distances/classes have been
supplied by these photometric predictions, so predictive agreement cannot prove
distance accuracy, contamination or real-population recovery.

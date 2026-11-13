# Outcome-unopened final target predictions

Main identity: candidate3540-cohort response12/calibration12, frozen within-class
SED kernels/binary mixture, and32luminosity-bootstrap conditional population
fits with their registered evidence weights. Zero/point and equal-bootstrap-cut
views use the same cohort and corresponding conditional populations. A broad
cohort run, if made, is a separately named transfer control. It cannot replace
the frozen main identity after validation outcomes become visible.

Use only frozen field526/common3032 photometry and numerical reference-sidecar
curves, never reference labels, distances, parallax scores or membership outcomes.
Authenticate all products. Existing field assignment uses actual original sky
bounds; new cells receive IDs10000+cell. For a field absent from the fitted
population draw its local logits from global_logits+0.35*N(0,1), seed sequence
[320401,correction_id,assigned_field]. Every star in that field shares the same
population draw. Missing fields are not assigned mean theta and each population
draw is normalized separately before the fixed cut mixture. New photometry does
not update the already fitted population/calibration state weights.

Reuse authenticated native response12 curves at the previous and selected
reference-sidecar step, or evaluate the same model at a required finer step.
Apply the unchanged log-space likelihood luminosity operation and each fitted
class/radial prior on TRUE distance modulus. Start with the previous and selected
reference steps; bounded additional0.005/0.0025/0.00125 levels are numerical only.
All four views require successive joint probability change<0.003, quantile
change<0.03mag, normalized density/joint-bin consistency1e-8, and interleaved
population-half marginal change<0.005. Preserve every level and terminal failure.

Retain full marginal logPDFs, class-distance probabilities, per-state likelihood
bin integrals/normalizing coefficients and authenticated shared population indices
so joint products can be reconstructed. Native curves and class logPDF tails are
not truncated by probability cuts. These predictions remain conditioned on the
fixed response/calibration and finite bootstrap library; response ensemble
probes are separately named sensitivity checks, not silently incorporated priors.

A deterministic pilot uses the eight lowest SHA256('320411:identity') ranks,
without changing any object's scientific inputs; a pilot marker never stands
for full526/3032prediction completion. Full predictions and all controls must
be frozen before the single newreference evaluation. Applicable candidate
selection, sky/training support and numerical flags accompany the final domain
report; predictions outside that admission are explicitly transfer diagnostics.

# v2.2: fixed-model application and selected-distance audit

User authorization2026-11-10: latest corrected review in pro_review_fbb8ea0.md.
Preserve v2.0/v2.1 models, cached PDFs, statistics and protocols. Write only new
data/v2_2 products, compact manifests/v2_2 and figures/v2_2. Strict verified
native DR11 PSF admission/export. No agents, timers, full-catalog prediction,
model-family search or replacement deployment.

1. Inspect raw finite/range/endpoint probability values, permit clipping only
within1e-7 roundoff. Rebuild every reliability stratum from frozen probabilities;
assert binned counts equal evaluation denominators. Record changes to Brier,
and prove all0.5/0.8/0.95 decisions unchanged. Preserve original tables.
2. Compute expanded_A on all26000 weighted application PSF objects and reproduce
baseline_v1 using its actual frozen catalog route A/B/C and wide Sobol128 model.
Keep all full PDFs in chunks; compare same model on development and application.
No new routing decisions. Report count, color/magnitude/error/WISE/sky, PDF
width/modes/branch mass and evidence coverage. Evidence grid fixed before
scoring: r edges18/20/22/25,gr−.2/.3/.8/1.3/2,rz0/.3/.6/1/1.5,
max optical error.03/.1/.2, WISE states none/W1/W1W2. Evidence categories
0/1–19/>=20 development chosen references per cell; counts do not certify accuracy.
3. Evaluate actual selected objects atq>.5/.8/.95: full distance/reference
metrics, interval score/coverage/width, all populations, corrected FeH, colors,
brightness, errors, WISE and sky. Include Wilson risk intervals and sky-cell
resampling; population-shift decomposition. q>.95 is now a frozen prospective
candidate rule, not retroactively independently validated.
4. Inventory identity-only unused objects separately from novel buffered sky;
keep prior5-ID reservation unchanged. Exclude all old train/eval/performance IDs,
historical/quarantined roles; log same-sky usefulness and missing system/external
distance evidence. Reserve without evaluating their outcome predictions.
5. Noise transfer uses at most256 high-quality development templates, disjoint
from each corresponding frozen model's training, no training templates.
Match actual weighted parent noise by observed gr/rz, target r strata18–20,
20–22,22–25 and WISE availability. Explicit bounded matching radius0.3mag
in(gr,rz); unmatched cases excluded and counted. Common-mode dimming changes
reference mu and all band magnitudes together. Require target band errors >=
template errors; inject sqrt(target²−template²), preserve source/base role.
Four fixed shared Gaussian replicates per identity/target. Evaluate full actual
Sobol128 distance PDF and the frozen optical class probe with and without
averaging the same128 error draws; no new classifier/calibration fit.
Report per-replicate and template-cluster variability, not independent-replica
sample sizes. This is conditional noise transfer, not new faint-population truth.
6. Far10–20/20–50: retain per-reference branch/window evidence, observable-cell
class-frequency versus known-class-conditioned contribution, train/cal/dev/app
selection differences, unlabeled training luminosity support and extreme Gaia
failure evidence. No claim of independent branch-shape truth from shared teachers.

Only if target-population calibration evidence and stable conditional shapes
justify it would another narrowly scoped candidate be warranted. This stage
first delivers diagnostics; no unvalidated reweighting, threshold tuning or
reuse of the rejected class-partition candidate. Science gaps remain explicit.

All full scans/predictions on allocated Slurm nodes; atomic checkpoints, source
hash locks and no duplicate submissions. Figures English, current Pal5 style;
scientific statistics use no display clipping. New tests specifically cover
endpoint inclusion and conservation, not merely mirrored implementation.

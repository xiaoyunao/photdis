# v1.4 scientific diagnostics protocol

Authorized 2026-11-05 by the user through the latest GPT Pro review of 46cb68a
(archived in pro_review_46cb68a.md). Baseline source commit: 46cb68a.

## Scope

Eight diagnostic groups: distance comparisons; reference/predicted Legacy-g CMD;
color residuals; sky residuals/footprints; spectral and observational dependencies;
probability calibration and fixed PDF examples; paired gains/losses with sky-block
uncertainty; weighted application support and WISE joint rejection conditions.
No fitting, hyperparameter changes, cuts for adoption, full-catalog application,
automations or subagents. Original references, models and evaluations are read-only.
All large-table joins, binning and limited forward prediction run in Slurm.

## Definitions and comparisons

- Main scientific rows are split=2, with the original model-specific eligibility.
  Class -1 is unknown, 0 dwarf, 1 evolved, 2 other/transition; never relabel unknown.
- Delta mu = predicted median minus reference modulus, distance in kpc:
  d=10**((mu-10)/5). Severe error means abs(delta mu)>1 mag. Report median,
  NMAD=1.4826 median absolute centered residual, 68/95% interval coverage and width.
- Gaia high-SNR anchors, SpecDis and SPDist are separate reference columns.
  Chosen reference follows frozen Gaia-then-SpecDis precedence. No weak/negative
  parallax inversion; full observed-parallax PIT/log scores use refined sidecars.
  VAC point-reference coverage measures agreement, not noise-free truth; include
  reference sigma_mu distributions and plot errors on fixed representative stars.
- Main physical plots show expanded_A (broad grz) and matched_griz (i-usable).
  Export metrics for all eight frozen/expanded/matched configurations. Compare
  A/B/C within old/new pairs only; paired grz/griz uses identical identities.
- Use reliable RVS parameter quality for continuous spectral panels. Metal-poor
  means [Fe/H]<-1.5 in this diagnostic, not a confirmed halo membership label.
- Spatial maps use original 10-degree RA/Dec cells; no interpolation. Show counts
  and per-square-degree density separately. Performance requires N>=20 per cell.
  Color bins require N>=10. Sparse cells remain blank, counts remain visible.
- Keep full distance/residual extents. Quantile-restricted color/parameter zooms
  must disclose excluded counts; complete metrics and extreme tails remain saved.
- Fixed plot bins: g-r [-1,3], r-z [-1,3], z-W1 [-2,6], r-i/i-z [-1,2];
  r [14,16,18,20,22,25], optical sigma [0,.01,.03,.1,.3,1], FeH [-4,-2,-1.5,-1,-.5,0,.5,1].
- Pair improvement = abs(new residual)-abs(old residual), negative means better.
  Save 2x2 severe-error transition counts by class. For tail-rate difference and
  mean absolute-error difference use 2000 paired whole-cell bootstrap replicates,
  seed20261105, percentile95% intervals; recompute star-weighted statistic after
  resampling cells. Report cell count and do not extrapolate to unlabeled sky.
- Fixed PDF examples: two distinct stars per group (ordinary success, evolved
  failure, ambiguity-flagged, no-W1, unknown), ordered by deterministic identity
  hash seed20261105, no visual cherry-picking. Selection uses existing outcomes
  only for explanatory case studies, not model evaluation. Predict the saved
  grz/griz pair only, with original Sobol128; require reproduced quantiles within
  1e-6. Retain full PDFs, reference constraints, near-reference probability mass,
  local peaks and support flags. Cases are not population frequency estimates.
- Parent coverage uses original26000 probabilities/weights; it represents the old
  southern DECam parent only. Recover raw WISE by exact native LSID from retained
  extraction checkpoints; audit duplicates/missing identities and measurement
  agreement. Joint SNR x WISEMASK x FRACFLUX includes overlapping failures,
  including low-SNR within the quality-failure group. No sequential-gate inference.

## Validation and delivery

Identity-safe one-to-one joins, reference/model/evaluation/refined hashes,
counts and consequential metrics must reconcile to frozen scores. Check bootstrap
sign, clustered resampling, unit transforms, empty/sparse bins and WISE overlap
with synthetic tests. Keep all figures and compact supporting statistics, with
source paths/hashes, reference counts and figure-specific interpretation. Inspect
rendered figures before delivery. Update WORKLOG/PLAN and sync all three endpoints.
These diagnostics make existing test results explanatory/development evidence
for any subsequent model changes; a later modified model needs new independent
final validation. Current deployed v1 remains unchanged.

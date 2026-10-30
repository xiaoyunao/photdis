# Lower RGB v2 controls (registered before new fitting)

Source: [Pro's complete review of 73b023a](pro_73b023a_review.md).
Question: how much of the apparent weak luminosity constraint comes from
training-domain leakage, curve shape, residual width and the bounded prior?

## Fixed design

- Freeze all v1 coefficients, data identities and scores. NGC 5466 is now an
  unblinded explanatory diagnostic, never a new independent validation.
- Read existing grz-clean M3/M53 lower-RGB members only. Determine common
  half-mag support cells from nominal labels FIRST (five per cluster);
  restrict training membership to those cells before any scale perturbation.
  Record selected identities and counts. Both gr and grz use these same rows.
- Equal total weights for the two retained clusters. No NGC 7006/upper RGB
  contribution to the lower fit. No tighter photometric/membership cuts.
- Compare the original quadratic plus additive FeH term against monotone
  piecewise linear color curves with four fixed Mg knots .5, 1.25, 2, 3 and
  one additive FeH term. Require nonpositive color slopes in all segments.
  Keep FeH ridge 4 and residual-color eigenvalue floor .01 mag. No knot search.
- Reuse the v1 M3/M53 distance, FeH and dust scale scenarios and .01-mag
  per-band zero-point scenarios. Generate 48 identical cluster-coherent
  nuisance realizations for ALL models/bands, including nominal draw zero;
  save the draw arrays, hash and same-source hash. Membership stays nominal.
- Inspect binned development residuals, derivative roots, covariance eigenvalues,
  and fraction of weighted residual trace contributed by the largest 1%.
  If this fraction exceeds 20% for the monotone grz nominal model, add one
  fixed robust likelihood control (specified before execution in its addendum).
  No star is rejected on distance residuals. Residual covariance includes
  model error and unmodeled population structure; it is not intrinsic width.
- Use the same supported grid, uniform luminosity prior and fixed abundance
  mixtures as v1. Known NGC 5466 FeH remains -2.16±.23; nominal development
  endpoints -2.00/-1.46 mean the diagnostic includes extrapolation.
- On the same original 93 supported NGC 5466 lower-RGB sources, report exact
  prior-only uniform[.5,3) baseline (median1.75, W68=1.70, logp=-ln2.5),
  quadratic and monotone paired gr/grz, and historical v1. Include Mg bins,
  center, NMAD, tails, coverage, widths and predictive log density. Score
  differences are NOT per-object KL divergences. No diagnostic-based tuning.
- Generate/recover: (i) mean-curve inputs at fixed luminosities, (ii) independent
  Gaussian color scatter plus bandwise errors, (iii) shared scale uncertainty.
  Distinguish deterministic shrinkage from proper generative coverage. Use
  12 simulated clusters with 100 stars each; sample a model/scales once per
  cluster, not once per star. Preserve the shared g error in color and final
  distance. Report per-cluster metrics and uncertainty by resampling clusters.

## Interpretation updates

NGC 7006 E(B−V)=.08±.05 is based on 41 RRab minimum-light colors and the
paper attributes the large uncertainty mainly to noisy light curves;
[section 4.2.3](https://arxiv.org/html/2212.05021v1) also quotes map uncertainties
.003/.004. The old .05 nuisance is a conservative stress scenario, not a
measurement of the SFD systematic error. Map formal errors are not total
systematic budgets either. M3/M53 v1 dust scenarios are retained as scenarios.

The old 0.083-mag paired gain compares two frozen fits with different Monte
Carlo nuisance draws. It does not isolate the z passband contribution exactly.
The discrete unknown-FeH mixtures omit the NGC 5466 nominal center: those
results test specified population priors, not generic loss from missing spectra.
A model with unsupported upper-RGB training weight and lower-RGB reversals
cannot establish a physical information limit. Assess that after these controls.

## Completion

Run real-table calculations on allocated NERSC compute. Freeze fitted models
before the post-unblinding diagnostic. Keep data on NERSC; sync code, protocol,
summary figures/results and final report. No new external system is needed for
these explanatory comparisons. A future claim of external predictive validity
requires a newly registered independent system.

### Fixed robust addendum (before viewing new development fits)

If the >20%-of-trace trigger is met, fit one contamination-mixture control:
95% narrow Gaussian and 5% Gaussian with 25 times the narrow residual
covariance, plus the actual measurement covariance in BOTH components.
Fit monotone means and positive-definite residual covariance jointly by
likelihood; keep the .01-mag eigenvalue floor, fixed mixture weights/scales,
FeH ridge and all 48 paired nuisance realizations. This is an exact declared
Gaussian-mixture working likelihood, not a Student-t approximation. Keep all
stars. No mixture-weight or broad-scale search. In synthetic recovery, draw
independent star mixture identities with those same fixed weights.

In shared-calibration recovery, draw one fitted calibration realization per
synthetic cluster: each realization already contains coherent developer
mu/E/FeH/g/r/z scale draws. The inference averages the same declared ensemble.
This tests uncertainty in the calibrated color relation, not an additional
unmodeled target-cluster dust systematic. Report this scope explicitly.

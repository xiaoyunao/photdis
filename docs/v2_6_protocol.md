# v2.6: observed-catalog posterior and response uncertainty

Frozen before production, 2026-11-11. Scope: full review `pro_review_39e6e4f.md`.
Preserve v2.5 fitted models, catalogs, grids and all negative evidence. No W1
experiment, prospective outcome opening, full-catalog prediction or deployment.

## A: estimands and shared-parameter inference
For cell b, r_j(theta)=eta_j R_jb/sum_k eta_k R_kb. Integrate using the
existing posterior, and draw ONE shared theta per latent-catalog realization.
Export 20-component per-star marginals, joint count draws/covariance, and
distance/class intervals. Compare counts with realized latent counts, separately
from selected population fractions versus design truth. Reuse all 192 catalogs.
Use 8192 posterior draws, first/second-half mean and endpoint diagnostics; record
Monte Carlo error. A repeated-population predictive interval is not this interval.

## B: restricted failure controls
Only reference_like, ordinary_dominated and giant_rich; entrance only, 12 catalogs
each, N=128. Compare (i) exact known-response cell draws, (ii) same cal-template
distribution with independently sampled native noise, (iii) existing independent
cal-to-test transfer. All use the original four-parameter prior [-3,3]^4.
Grid 2^18; nested half-grid endpoint comparisons. Independent test response is
used ONLY for truth. Reachability bounds use the fact that at fixed distance tilt,
distance fraction is linear-fractional in the three group odds: extrema occur at
their eight corners. Optimize the remaining bounded scalar tilt and report both
grid and optimizer extrema, with numerical (not formal interval-arithmetic) status.

## C: finite sources versus response simulation
Nested identity-hash cal quotas [64,64,48,48,32], [192,192,144,144,96],
[512,512,384,384,256] for ordinary/poor/evolved/other/missing. Strict PSF,
same eligibility and test exclusion as v2.5; no outcome-based choice of scale.
Per-source responses use 256 independent native-noise draws per distance; retain
the first 128 as a Monte Carlo check. Missing-label membership uses the same
fixed cal color/Mg kernel. Sources, not noise replicates, are bootstrap units.
Largest scale: 16 exponential-weight Bayesian-bootstrap response draws define an
explicit finite empirical prior on responses. Joint posterior p(R,theta|Y) uses
the selected-cell likelihood and EVIDENCE weights, not equal normalized averages.
Theta grid 2^16 with nested-half diagnostic; compare first 8 versus all 16 response
prior draws. This finite response prior and its convergence limits must be stated.
Export response posterior ESS, parameter ESS, counts and intervals for the same
36 existing entrance catalogs. No acceptance just because intervals widen.
Inspect cal/test poor/other color-Mg summaries and component observation residuals.

## D: one source-conditioned poor luminosity calibration
Use cal-role PSF signed Gaia observations only to constrain a poor-branch
luminosity offset and extra dispersion relative to VAC teacher distances.
SpecDis and SPDist are separate source-conditioned experiments, with an ordinary
calibrator zero-point as a source-scale nuisance. No Gaia-inverted teacher or
positive-SNR cut. Fixed offset grid [-1,1] mag and scatter [0,.6] mag; integrate
reference distance uncertainty and full signed Gaia likelihood. Reference selection
is retained, not claimed ignorable or representative of the photometric parent.
Exclude evaluation identities and ALL reused-control group members from cal.
Evaluate the original paired identities with the same source qualification before
and after; full likelihood/PIT and predictive intervals, and clustered paired
changes. These source-conditioned checks alone are not photometric validation.
Also propagate the calibrated poor-component luminosity uncertainty through the
fixed empirical grz template mixture, with unknown metallicity kept latent and
all four components retained. No hard spectroscopic branch at application.
This is an isolated prototype; freeze its numerical choices before production.

## Delivery and acceptance
Independent tests of responsibilities, shared count covariance and response
evidence; source/identity/PSF audits and frozen-input hashes. English-only Pal5
style PNG300dpi/PDF plus compact CSV/JSON, visually inspected. Report failures,
finite-library dependence, teacher uncertainty, response-prior discretization,
middle-metallicity gap, five distance nodes, and no new blind validation.
Keep 3250 prospective identities unopened. Update WORKLOG/PLAN and safe git sync.

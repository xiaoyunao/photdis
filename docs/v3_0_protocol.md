# Photdis v3.0 development contract

Status: implementation in progress, not a scientifically accepted model or release.
Authoritative scope: [complete Pro review](pro_review_b59e18b.md). This contract
records choices before production; final E-stage freeze will separately hash all
model/calibration/prior/metric artifacts before prospective outcomes are opened.

## A. Evidence and roles
Native TYPE=PSF is mandatory at admission/export. Exact int64 identity, native
ls_id and source row provenance are retained; ambiguous identities are excluded
from fitting and reported. No class/FeH label is required for distance evidence.
Missing labels, posterior ambiguity, and model inadequacy are distinct states.

Initial reference inventory uses frozen v1.4 full reference identities plus the
v2.0 all-reference native report (not only chosen distances). Raw SpecDis rows are
rejoined without BINARY_FLAG selection. Spectral quality excludes the invalid-
parameter bit from its distance-only gate; absent physical labels remain absent.
Preserve all existing distance/astrometric qualifications and raw measurements.
A primary evidence channel is fixed by availability: qualified SpecDis, otherwise
qualified SPDist, otherwise signed Gaia likelihood. Never multiply the two VACs
or their Gaia input as independent evidence. Alternative channels are sensitivity
and consistency checks. This is a source-conditioned research reference, not an
independent absolute truth set. Source-calibration modeling must account for this.

The old native clean grz S/N>=5, |b|>30, 0<=E(B-V)<0.03, DECam release11010
reference envelope is the starting inventory, with separate exclusions. Optional
i/WISE use all legal fluxes, including negative/low S/N values; artifacts and
missing observations are distinguished. No good-WISE admission requirement.
The exact scientific and application envelope will be fixed on development before E.

Use previous source identities and sky partitions to prevent cross-role reuse:
SED train, source calibration, hyperparameter selection, seen development, final
real validation. Historical fitted objects cannot enter a calibration/evaluation
role. Source calibration and hyperparameter selection split calibration sky cells
by a fixed hash; previously examined samples remain explicitly development.
Original 3250 reservation stays closed until E. Its selection is conditional on
13 already-used fields/historical exclusions. New sky selection must be frozen
from photometry before reference association, with no-reference objects retained.

## B. Native-flux continuous inference
One canonical order: G,R,I,Z,W1,W2; explicit survey/filter version per observation.
DECam and BASS/MzLS are separate systems. No north i imputation or DECam aliasing.
Measure native nanomaggy flux once per band. Missing bands contribute no likelihood;
legal signed detections use Gaussian flux likelihood; actual upper limits use CDF.
Use raw flux and forward extinction; do not also deredden the observations.
Common dust/calibration perturbations are shared draw variables; per-object noise
is separate. Empirical SED intrinsic covariance must subtract modeled measurement
covariance, including shared distance errors, then be regularized / audited.

Distance modulus is continuous with adaptive quadrature, boundary extension and
mode checks. Output distance bins are integrals, not latent distance nodes. Keep
near/far modes and >50kpc tails. Preserve reweightable joint latent support and the
actual reference joint prior. Empirical SED support is not a prior-free likelihood.
Finite SED kernels, correlated widths, luminosity tails, contiguous metallicity,
turnoff/subgiants, missing labels, brighter binary and constrained outlier channels
are included. Application inference cannot use spectral truth or target Gaia.

## C. Calibration, population and counts
Source offsets, excess scatter and reference heavy tails are distinct from the
physical metallicity-luminosity relation. Shared source/SED/calibration uncertainty
must propagate through per-star posteriors and counts. Limited branch-specific
continuous-distance and within-branch SED weights, regularized across sky cells,
replace unrestricted per-template weights. Selection normalization is explicit.
Deliver both fixed reference-prior and target-population joint posteriors.

For a realized catalog sum joint P(class,d-bin), never multiply marginal class and
distance probabilities. Conditional count covariance is sum(diag(p_i)-p_i p_i^T);
add covariance of shared conditional count means. Use int64 realized counts.
Final marginal MC precision is measured after evidence-weighted marginalization,
not the maximum precision of an almost-zero-weight conditional response.
Observed-catalog counts are not an intrinsic Galactic density. Sampled parent
counts additionally carry design uncertainty; full-catalog products use unit weights.

## D–F. Evaluation and actual delivery
D: same identities/roles for QRF-grz versus v3-grz, v3-grz versus multiband, and
reference versus target prior. One fixed probabilistic classifier probe compares
no-color (magnitude/sky/errors), colors plus optional bands, and distance-marginalized
v3 class probabilities; log score, Brier, reliability and class-specific results.
No Mg/reference-distance input, no mutual-information fraction claims. Continuous
simulations include independent SEDs, off-node distances, middle metallicity,
class-distance correlation, narrow structures, binaries, weak/missing WISE, dust
and system shifts. Retain known/same/transfer regression controls. Both distance
and class counts, widths and joint probabilities must be assessed.

E: freeze method, priors, metrics and numerical thresholds before once-only
reservation evaluation; new photometric-first sky and available independent scale.
Pure-photometry full signed Gaia predictive density includes negative parallax.
Report bias, NMAD, catastrophic errors, scores, 68/95 coverage and width, both PIT
tails and counts by sky/magnitude/color/source/bands. Mandatory failure strata:
2–5kpc, red, metal-poor, other. Initial refinement target |delta qnear|<0.005;
confirm on development and freeze explicit integration/count-MC tolerances.

F: benchmark measured time/memory/output first, then chunked existing-field
qualified candidate catalog. Candidate pruning/compression must preserve secondary
modes against full computation on a fixed subset. Chunks share global draw IDs.
Star products: exact IDs/system/band mask, quantiles/modes/masses, class-distance
joint probabilities and qnear/far bins, joint reweightable support/reference prior,
support/extrapolation/multimodal/prior-sensitive flags and all version hashes.
Shared parameter products, near/far group counts/covariance, predicted versus
independently measured contamination, one real marginalization science example,
English Pal5-style PNG/PDF with inspectable data, report/atlas/requirement ledger.

Engineering completion and scientific acceptance are separate: mark each domain
supported, limited, or unverified. Preserve failures and old deployed v1. No model
family search, subagents, timers or automatic deployment. Large compute on Slurm;
large data stay NERSC. Update WORKLOG/PLAN/git at actual milestones.

## Source documentation verified 2026-11-11
- [DESI SpecDis](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-specdis/):
  v2.1 distances in kpc; neural spectral distances trained on Gaia. Binary flag
  derives from the difference between neural and Gaia geometric absolute G.
- [DESI SPDist](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-spdist/):
  Gaia photometry and DESI atmospheric parameters; retain input-quality flags and
  documented intrinsic magnitude scatter. Not independent of Gaia/SpecDis by name.
- [Legacy DR11](https://www.legacysurvey.org/dr11/description/): new DECam griz
  reductions; northern imaging remains the separate DR9 BASS/MzLS system.

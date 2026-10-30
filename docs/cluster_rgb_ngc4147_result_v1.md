# Independent NGC 4147 lower-RGB transfer

**The new cluster retains useful color information, but its distance scale
is sensitive to the reference system.** The preregistered robust monotone
grz model has median Δμ=−0.205 mag (9.0% nearer), NMAD=0.187 mag and
68/95% coverage 29/42 and 37/42. Its average log predictive density improves
by 0.371 per star over a uniform luminosity prior. The source-paired z gain
is small and uncertain on this cluster. Keep the same frozen primary as a
conditional research model; the result motivates resolving shared scales
before adopting a field RGB distance distribution.

## What was fixed before the test

The [protocol](cluster_rgb_ngc4147_protocol_v1.md),
[configuration](../configs/cluster_rgb_ngc4147_v1.json) and model hashes were
committed in `0a8b940`, before extraction. Coverage, final IDs, reference
sensitivities and scoring implementation were committed in `7768eb3`,
before scoring. The primary is robust monotone grz; other frozen fits are
controls. No model fitting or residual-based sample changes occurred.
Existing v1/v2 models and NGC 5466 diagnostics remain unchanged.

Earlier NGC 4147 use was a spectral-count ranking, not RGB model development.
Slurm 60535204 extracted 1,210 unique Gaia-linked members, all southern
DECam RELEASE=11010. Reconstructed development identities reproduce the
frozen hash and have zero overlap with these members. Membership P>0.9,
the copied Gaia-CMD lower-RGB region and existing band quality yield:

| Reference Mg | Clean gr | Common clean grz |
|---|---:|---:|
| 0.5–1.0 | 6 | 6 |
| 1.0–1.5 | 12 | 12 |
| 1.5–2.0 | 11 | 10 |
| 2.0–2.5 | 9 | 6 |
| 2.5–3.0 | 10 | 8 |
| Total | 48 | 42 |

Thus 14 common objects test Mg≥2. These are RGB candidates selected by
Gaia phase evidence. The domain cut uses the external reference luminosity
for validation. Exact IDs, catalog hash and counts are in the
[coverage lock](../manifests/cluster_rgb_ngc4147_blind_coverage_v1.json).
All comparisons below use the same 42 objects. The six gr-only additions
are separately scored: center −0.073, NMAD 0.162 mag, all six inside the
68% interval; they do not enter the same-star z comparison.

## Reference abundance and fixed-model comparisons

The [2018 RRc analysis](https://arxiv.org/abs/1803.05313) provides
19.05±0.46 kpc, hence μ=16.3995±0.0524 mag. Its ZW abundance transforms
to −1.6898±0.2032 on the development UVES scale. RRc is used because the
RRab Fourier calibrations are unsuitable for this cluster. Target
abundance uncertainty is marginalized with the frozen three-node rule.
Δμ is predicted minus external modulus throughout; widths are central
posterior widths and coverage is against measured g0−μ.

| Model | Median Δμ | NMAD | Tail >0.5 | 68% cover | 95% cover | W68 | W95 | Mean log density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior only | −0.144 | 0.665 | 50.0% | 73.8% | 100% | 1.70 | 2.375 | −0.916 |
| Quadratic gr | −0.170 | 0.219 | 14.3% | 81.0% | 95.2% | 0.97 | 1.85 | −0.586 |
| Quadratic grz | −0.225 | 0.172 | 16.7% | 73.8% | 90.5% | 0.80 | 1.58 | −0.880 |
| Monotone gr | −0.160 | 0.210 | 14.3% | 81.0% | 95.2% | 0.99 | 1.87 | −0.588 |
| Monotone grz | −0.225 | 0.178 | 16.7% | 73.8% | 92.9% | 0.79 | 1.57 | −0.872 |
| Robust gr | −0.140 | 0.210 | 14.3% | 78.6% | 95.2% | 0.80 | 1.75 | −0.398 |
| **Registered primary robust grz** | **−0.205** | **0.187** | **19.0%** | **69.0%** | **88.1%** | **0.66** | **1.41** | **−0.546** |

Centers, scatter and widths are in mag; tail=|Δμ|>0.5. The primary's
W68 is 39% of prior-only and implies distance-endpoint ratio 1.36.
It improves the point-score over prior-only but has a worse score and
larger tail than robust gr here. The robust gr-minus-grz NMAD gain is
0.022 mag, paired-source bootstrap 95% interval [−0.047,+0.134]. Ordinary
quadratic and monotone gains also have intervals containing zero. This
sample does not establish an additional z gain; the primary was not
reselected after seeing the comparison. Bootstrap conditions on this
cluster and frozen calibration; it is not a population-of-clusters error.

See [the actual CMD and same-star residual figure](../figures/cluster_rgb/08_ngc4147_external_v1.png).
All objects, including large residuals, are retained.

## Luminosity dependence and matched recovery

| Reference Mg | N | Primary center | NMAD | W68 | 68% / 95% cover |
|---|---:|---:|---:|---:|---:|
| 0.5–1.0 | 6 | −0.129 | 0.082 | 0.55 | 4/6 / 4/6 |
| 1.0–1.5 | 12 | −0.297 | 0.136 | 0.68 | 7/12 / 10/12 |
| 1.5–2.0 | 10 | −0.282 | 0.105 | 0.71 | 8/10 / 10/10 |
| 2.0–2.5 | 6 | −0.201 | 0.126 | 0.59 | 4/6 / 6/6 |
| 2.5–3.0 | 8 | +0.034 | 0.172 | 0.55 | 6/8 / 7/8 |

The faint-minus-1.0–1.5 center contrast is **+0.330 mag**. The target-matched
generator has three fixed modes, 48 synthetic clusters each:

- A fixed grid of 80 stars, nominal target/calibration scales, empirical
  per-Mg-bin photometric error vectors, mixture residuals and band noise.
- The same grid with one calibration realization and one μ, E(B−V), FeH,
  and g/r/z zero-point draw shared by the entire synthetic cluster.
- The observed 42-member reference-Mg distribution with the same shared
  treatment. This plug-in design matches the sampled luminosity distribution.

Inference always uses the frozen ensemble. Actual error vectors are drawn
within the relevant half-mag bin. The same noisy g enters colors and final
distance modulus. Simulations retain both true-Mg and noisy-reference-Mg
bins, including out-of-support noisy references in distance/coverage
statistics. Full log density is undefined for such boundary crossings;
interior scores are separately recorded. These are conditional RGB
simulations, not a model of the Gaia phase-selection function.

Under nominal scales the grid predicts median bin residuals of about
−0.132, −0.061, −0.082, −0.056, +0.136 mag: bounded inference naturally
produces a faint-end rise. Its median faint–bright contrast is 0.198 mag.
For the actual-size, shared-scale simulation the contrast is **0.299 mag**,
with a 95% whole-cluster predictive range **[0.039,0.768]**, compared with
the observed 0.330. This contrast is formed *within each realization*, so
bin correlations and common shifts are preserved. Its true-Mg-bin version
is 0.297 [−0.017,0.772]. The larger nominal-grid sample is a shape control,
not an equal-count significance test of the 42 real objects.

The observed trend is compatible with this registered model plus shared
scales. These broad ranges do not isolate a precise excess trend. Forty-eight
cluster draws give a diagnostic predictive envelope; endpoints are Monte
Carlo estimates, not precise tail calibration. In shared simulations 13/48
grid and 8/48 observed-design FeH draws cross the development-center range
[−2.00,−1.46]; the fixed linear FeH continuation is an additional modeling
assumption in these envelopes.

The [recovery figure](../figures/cluster_rgb/09_ngc4147_recovery_v1.png),
[per-cluster draws](../manifests/cluster_rgb_ngc4147_recovery_v1.json) and
[correlated-contrast summary](../manifests/cluster_rgb_ngc4147_summary_v1.json)
preserve the full comparisons. The contrast summary is derived from saved
draws after scoring; it changes no inference or selection.

## Shared scales and abundance alternatives

The actual-size shared generator has cluster-center median −0.051 mag and
95% predictive range **[−0.571,+0.479]**. The measured −0.205 lies inside.
This compatibility is conditional on the registered scale distribution;
the offset itself remains measured. The external modulus error alone is
0.052 mag and cannot account for its full size at one sigma. No shared
uncertainty has been divided by sqrt(42).

On the same IDs, the independently registered reference changes give:

| Sensitivity | Primary median Δμ |
|---|---:|
| Nominal RRc/Legacy references | −0.205 |
| Alternative I-band RRc distance 18.89 kpc | −0.187 |
| Compilation RRc distance 18.7 kpc | −0.165 |
| Additional E(B−V)=−0.01 / +0.01 | +0.007 / −0.425 |
| Coherent g zero point −0.01 / +0.01 mag | −0.313 / −0.114 |
| Coherent r zero point −0.01 / +0.01 mag | −0.199 / −0.225 |
| Coherent z zero point −0.01 / +0.01 mag | −0.125 / −0.285 |
| Shift Legacy median E(B−V)=0.02574 to 0.01 | +0.119 |
| Shift Legacy median E(B−V)=0.02574 to 0.02 | −0.087 |

The final two cases retain differential reddening and shift the whole
cluster median to the predeclared literature values. They show leverage
of the reference scale, not which reddening estimate is correct. The 2018
I-band estimate uses the same RRc stars; it is not combined as an independent
distance constraint. The [2023 compilation](https://arxiv.org/html/2306.01175v1)
has E(B−V)=0.01 in the abundance table and 0.02 in the distance table.
Distance/abundance covariance from the RRc reference is unavailable; the
shared generator assumes independence and the scenarios remain explicit.

| FeH condition | Median Δμ | NMAD | W68 | 68% / 95% cover |
|---|---:|---:|---:|---:|
| Primary RRc with uncertainty | −0.205 | 0.187 | 0.66 | 69.0% / 88.1% |
| Exact primary center | −0.224 | 0.183 | 0.54 | 57.1% / 81.0% |
| RRc compilation −1.68, propagated wider scatter | −0.209 | 0.200 | 0.72 | 73.8% / 92.9% |
| Spectroscopic −1.84, registered scale floor | −0.105 | 0.180 | 0.62 | 76.2% / 88.1% |
| Unknown FeH, rich-heavy | −0.305 | 0.211 | 0.72 | 57.1% / 88.1% |
| Unknown FeH, equal mixture | −0.183 | 0.183 | 0.79 | 78.6% / 92.9% |
| Unknown FeH, poor-heavy | −0.065 | 0.145 | 0.78 | 76.2% / 92.9% |

The [spectroscopic mean](https://arxiv.org/abs/1605.03408) has a reported
0.02-dex mean statistical error. This sensitivity adds a prespecified
0.10-dex systematic floor in quadrature; it does not assert an exact
UVES cross-calibration. The unknown mixtures retain the original −1.46
and −2.00 components. Changing their weights moves the center by 0.240 mag
(11.7% in distance ratio) despite similar scatter. None was chosen to
minimize the newly observed offset.

The frozen central model gives covariance-weighted local compensation
dMg/dFeH=0.784, 0.839, 1.080 across its three segments. The corresponding
color-derivative cosines are −0.9956, −0.9994, −0.9994. Thus abundance and
luminosity nearly compensate within this two-cluster relation. This is a
local model calculation, not a universal grz information limit. The fixed
5% broad component remains a residual model, not a measured contamination
probability or a classification of any of these sources.

## Decision and next step

The four registered questions now have measured answers:

1. **Scale:** the −0.205-mag center is compatible with the broad declared
   shared-scale generator, while its nominal offset is substantial.
2. **Trend:** the 0.330-mag contrast fits the matched generator's correlated
   range; present data do not separate an extra trend from shared-scale
   response and bounded inference.
3. **Color information:** the primary improves on prior-only in scatter,
   width and point predictive score. An additional z benefit is uncertain
   here; robust gr actually scores better in aggregate.
4. **Unknown abundance:** prior weights move the whole scale by 0.240 mag.

This completes one independent conditional transfer test. Retain its
negative results and primary identity. The immediate useful follow-up is
an external reference-scale comparison on these systems—particularly the
distance-independent reddening/photometric zero points and abundance scale—
followed by population-prior or shared-member inference once those scales
are constrained. The current result does not motivate another polynomial
or mixture-weight search. Preserve all existing upper-RGB and cold-dwarf
branches as they are.

## Reproduction and validation

NERSC extraction Slurm **60535204**, scoring/recovery **60535316** both
completed with exit 0 (scoring/recovery took 2m02s). Programs are
`scripts/run_cluster_rgb_ngc4147_blind_extract.sh` and
`scripts/run_cluster_rgb_ngc4147_score.sh`; the small derived summary is
`PYTHONPATH=src python3 scripts/cluster_rgb_ngc4147_summary.py`.
Source catalogs and scores stay under NERSC `data/cluster_rgb_ngc4147_v1`.
The [score manifest](../manifests/cluster_rgb_ngc4147_result_v1.json) records
config, coverage, model and identity hashes. Both model manifests still
match the preregistration. All 21 focused tests pass, covering shared-g and scale signs,
empirical error-vector matching, source uniqueness, hash immutability,
quadrature, analytic prior, and correlated whole-cluster summaries.

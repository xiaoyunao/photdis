# Lower RGB scope, shape and prior controls

The main finding is a modeling correction: **restricting training to the
stated lower-RGB domain removes the old color reversals even with the same
quadratic form**. The old broad posteriors did not establish a grz physical
information limit. This experiment follows the [registered protocol](cluster_rgb_lower_protocol_v2.md)
and [full Pro review](pro_73b023a_review.md). Frozen v1 scores are unchanged.

## Training-domain and shape audit

The [v1 audit](../manifests/cluster_rgb_v1_scope_audit.json) reproduces 850
training members, 672 in lower common support, and 46.05% of pre-covariance
weight attached to upper RGB. Its gr/rz curves reverse within lower support
in 48/48 and 45/48 realizations. In v2 support is computed **before fitting**:
M3 contributes 422 and M53 250 common grz-clean lower-RGB members. The same
672 identities, Mg cells [0.5,3), and 48 paired cluster-scale realizations
feed both bands and both forms. Selection remains fixed under scale draws.
NGC 7006's upper RGB contributes no weight to these lower fits.

The restricted quadratic and fixed four-knot monotone fits have no positive
color slopes in any of their 48 realizations. Their curves and binned
residuals are very similar. The brightest M3 half-mag bin has only five
members and a gr median residual of about 0.02 mag; it remains visible in
[the development figure](../figures/cluster_rgb/07_lower_rgb_development_v2.png).
The fit identity, nuisance-draw and input hashes are in the
[frozen model](../manifests/cluster_rgb_lower_model_v2.json).

The largest 1% of monotone-grz development residuals contribute 30.3% of the
cluster-weighted residual trace, exceeding the predeclared 20% trigger.
The fixed robust control retains every star and fits an exact 95% narrow /
5% broad Gaussian residual mixture; broad intrinsic covariance is 25 times
narrow, while measured errors are added to each component separately.
Its narrow color scales are 0.0139/0.0156 mag, versus 0.0238/0.0201 for the
ordinary monotone fit. These are effective model residual components, not
measurements of pure stellar intrinsic width. The [robust model](../manifests/cluster_rgb_lower_robust_v2.json)
records the original parent hash and the same paired training draws.

## The original 93 NGC 5466 members, now an explanatory diagnostic

These results use exactly the v1 common identities. NGC 5466 is already
unblinded; no node, quality cut, model variant or prior weight was chosen
from its new score. Here delta_mu means predicted minus external modulus.
The single external reference has its shared 0.09-mag uncertainty.

| Model | Median delta_mu | NMAD | Tail >0.5 mag | 68% cover | 95% cover | Median W68 | Mean log density |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform luminosity prior only | -0.139 | 0.583 | 38.7% | 88.2% | 100% | 1.70 | -0.916 |
| Frozen v1 grz | -0.011 | 0.231 | 9.7% | 95.7% | 100% | 1.50 | -0.639 |
| Lower quadratic gr | +0.040 | 0.177 | 4.3% | 95.7% | 97.8% | 1.18 | -0.604 |
| Lower quadratic grz | +0.063 | 0.125 | 3.2% | 95.7% | 97.8% | 0.96 | -0.361 |
| Lower monotone gr | +0.045 | 0.179 | 4.3% | 95.7% | 97.8% | 1.20 | -0.614 |
| Lower monotone grz | +0.068 | 0.130 | 3.2% | 95.7% | 97.8% | 0.98 | -0.370 |
| Robust monotone gr | +0.064 | 0.173 | 3.2% | 95.7% | 98.9% | 1.06 | -0.336 |
| Robust monotone grz | +0.043 | 0.110 | 3.2% | 96.8% | 98.9% | 0.84 | -0.089 |

The 0.110-mag NMAD is within-system point scatter in an exposed test with
reference abundance, not a universal 5% RGB distance error. Its 0.84-mag
68% width corresponds to a distance-endpoint ratio of 1.47; the faint-bin
+0.245-mag center is a roughly 12% distance offset. Shared system scales
are not represented by within-cluster NMAD. The 5% broad residual mixture
weight is prescribed, not a measured HB/AGB/nonmember/binary fraction.

Magnitude units apply to center, NMAD and width. The exact prior-only model
has median Mg=1.75 and central 68% width=1.70. Its 88.2% coverage already
shows this selected cluster sample is not uniformly distributed in Mg.
Robust grz narrows width to about 49% of that prior width and improves mean
log density by 0.827 per star; that score difference is not individual KL
information. Coverage on this one cluster remains well above 68%.

All band gains now use paired nuisance draws as well as identical sources.
The gr-minus-grz NMAD gain and paired-source bootstrap 95% interval are:
quadratic 0.052 [-0.009,0.086], monotone 0.049 [0.001,0.091], robust 0.062
[0.007,0.090] mag. These intervals condition on the frozen calibration
ensemble; they do not sample a population of independent real clusters.

### Luminosity dependence

| Reference Mg | N | Prior-only center | Lower quadratic grz | Lower monotone grz | Robust grz |
|---|---:|---:|---:|---:|---:|
| 0.5–1.0 | 6 | -0.892 | -0.082 | -0.082 | -0.052 |
| 1.0–1.5 | 34 | -0.459 | +0.006 | +0.025 | +0.004 |
| 1.5–2.0 | 25 | -0.059 | +0.022 | +0.022 | -0.020 |
| 2.0–2.5 | 19 | +0.417 | +0.161 | +0.161 | +0.105 |
| 2.5–3.0 | 9 | +0.915 | +0.318 | +0.318 | +0.245 |

The old 1–1.5 versus 2.5–3 contrast was 0.686 mag. It falls to about 0.312
with restricted quadratic and 0.241 with the robust control; a faint-end
trend remains. The [paired diagnostic figure](../figures/cluster_rgb/07_lower_rgb_controls_v2r1.png)
shows the contraction and the few large residuals, all retained.

## Internal generation and recovery

Each model has three registered experiments, 12 synthetic clusters of 100
stars: deterministic mean-curve inputs; independent residual scatter plus
0.003-mag band errors; then a shared calibration realization selected once
per cluster. The last realization comes from the fitted paired developer
mu/E/FeH/zero-point draws. It is uncertainty in the calibrated relation,
not additional independent target-cluster dust uncertainty. Inference
marginalizes the same declared ensemble. FeH=-1.7 is known in these internal
experiments, whereas NGC 5466 uses an uncertain, mildly extrapolated FeH.

The same noisy g enters colors and final distance; bins use true generated
Mg. The scatter generator is Gaussian for the ordinary models and the
registered Gaussian mixture for the robust model. Photometric propagation
uses the same local shared-band covariance approximation as v1. The
mean-curve run demonstrates contraction, not coverage under a full generator.

| grz model | Scatter+photometry 68% / 95% coverage | Shared calibration 68% / 95% coverage |
|---|---:|---:|
| Lower quadratic | 69.2% / 96.0% | 66.1% / 96.3% |
| Lower monotone | 68.3% / 96.0% | 66.1% / 96.5% |
| Robust monotone | 68.0% / 94.9% | 67.6% / 94.4% |

The shared-calibration robust 68% coverage has a **cluster-resampled** 95%
interval about 58.7%–75.8%. Twelve cluster realizations are sufficient for
this diagnostic, not a precise general coverage claim. Its pooled residual
center is -0.096 mag for these draws; cluster variability must remain visible.
Per-cluster and true-Mg-bin metrics are preserved in the result manifest.

A handful of simulated noisy-g point references cross the hard supported
Mg boundary (up to 5/1200). Their full log density is undefined under this
bounded-grid approximation. Revision v2r1 reports that fact and a separate
interior score, rather than averaging a numerical log floor. The revision
changes no fits, draws, selections or NGC 5466 results.

## Abundance and scale interpretation

Fixed unknown-FeH mixture weights shift robust-grz centers from -0.300
(rich-heavy), through -0.177 (equal), to -0.036 mag (poor-heavy). The known
FeH result is +0.043. These are specified-prior experiments: the mixture
centers -1.46/-2.00 omit NGC 5466's nominal -2.16. They combine population
prior mismatch, extrapolation and the color–abundance degeneracy.

The [NGC 7006 RR Lyrae paper, section 4.2.3](https://arxiv.org/html/2212.05021v1)
attributes its 0.05 reddening uncertainty mainly to noisy RRab light curves.
That v1 choice was a conservative stress scenario, not a measured SFD
systematic uncertainty. The new lower fit naturally excludes its influence.
The M3/M53 scenarios and residual floor remain as registered.

## Decision and next experiment

The training-domain mistake materially inflated the lower-RGB failure.
Fixing it removes the quadratic reversals and sharply improves the existing
cluster diagnostic. Monotonicity is a useful shape constraint but produces
little additional improvement here. The robust control shows that a few
large development residuals also influenced the old effective width.

Keep the three new controls frozen. The robust variant is a candidate for a
new independently registered system test, not selected as a field product
by this exposed cluster. Before adding bands or invoking a physical grz
limit, test it on a new same-system lower-RGB cluster with comparable FeH and
supported magnitude range. Register sample/stage definitions and nuisance
scenarios first, include prior-only, and evaluate the remaining faint-end
trend and abundance dependence. A wider giant catalog still needs independent
stage and population evidence; this round does not calibrate upper RGB.

Outputs: [all diagnostic metrics](../manifests/cluster_rgb_lower_diagnostic_v2r1.json),
[frozen ordinary models](../manifests/cluster_rgb_lower_model_v2.json),
[frozen robust models](../manifests/cluster_rgb_lower_robust_v2.json).

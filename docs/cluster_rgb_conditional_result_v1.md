# DECam conditional RGB v1: new whole-cluster transfer

The pre-registered experiment, frozen model and once-only NGC 5466 score are in [the protocol](cluster_rgb_conditional_protocol_v1.md), [fit report](cluster_rgb_conditional_fit_v1.md), [model manifest](../manifests/cluster_rgb_conditional_model_v1.json) and [blind score manifest](../manifests/cluster_rgb_ngc5466_blind_score_v1.json). Program, manifests and selected figures are tracked; member catalogs and per-star score tables remain on NERSC.

## Sample and support

The external [RR Lyrae measurement for NGC 5466](https://arxiv.org/abs/0711.4027) supplies the frozen true modulus `16.02 ± 0.09 mag`. The [homogeneous RR Lyrae abundance compilation](https://arxiv.org/html/2306.01175v1) supplies the registered RRab-to-UVES abundance `−2.16 ± 0.23 dex` for the known-abundance test; its RRc-derived `−1.89 dex` is a predeclared sensitivity. The existing [Vasiliev–Baumgardt membership archive](https://zenodo.org/records/4891252) supplies Gaia IDs and membership probabilities.

The Slurm extraction found 4,918 unique DR11 matches, all with southern DECam release `11010`. Among `Pmember>0.9` Gaia-CMD candidates, the upper RGB has 52 gr-clean and 32 grz-clean records; the lower RGB has 96 and 93. These are broad phase candidates, not spectroscopic confirmation. The registered overlap rule supports the upper RGB in gr only and the lower RGB in both gr and grz. It accepts 135 gr-only-model test stars and 93 stars for the exact same-source gr/grz comparison. Gr quality recovers 23 additional RGB candidates that fail grz quality, counted separately from the information-gain comparison.

## Whole-cluster result with known iron abundance

Residual is `Δμ = predicted distance modulus − independent RR Lyrae modulus`. Positive means predicted too far. The point reference has one shared 0.09-mag uncertainty for the cluster.

| Model and scored population | N | Median Δμ | NMAD | `|Δμ|>0.5` | 68% cover | 95% cover | Median 68% interval width |
|---|---:|---:|---:|---:|---:|---:|---:|
| gr all, supported upper and lower | 135 | +0.086 | 0.427 | 16.3% | 85.2% | 99.3% | 1.62 mag |
| gr on grz-common lower RGB | 93 | −0.109 | 0.314 | 12.9% | 94.6% | 100% | 1.62 mag |
| grz on the same 93 lower-RGB stars | 93 | −0.011 | 0.231 | 9.7% | 95.7% | 100% | 1.50 mag |

The grz model improves the same-source core scatter and center relative to gr. The paired NMAD improvement is `0.083 mag`; a source bootstrap on those 93 stars gives a 95% interval of `+0.009 to +0.140 mag` ([paired audit](../manifests/cluster_rgb_ngc5466_pair_audit_v1.json)). Mean held-out log density in distance-modulus space also improves from `−0.744` to `−0.639` per star. Its 68% interval contains nearly 96% of references and is very wide: the median interval spans 1.50 mag in modulus. This is an overconservative *conditional* interval on this one cluster, not a claim of calibrated 68% probability for field giants. The grz upper-RGB result cannot be evaluated: no upper common cell passed the registered two-cluster count threshold.

The overall grz center near zero conceals a strong luminosity dependence:

| Reference `M_g` bin | N | grz median Δμ | grz NMAD |
|---|---:|---:|---:|
| 0.5–1.0 | 6 | −0.262 | 0.036 |
| 1.0–1.5 | 34 | −0.131 | 0.082 |
| 1.5–2.0 | 25 | +0.019 | 0.070 |
| 2.0–2.5 | 19 | +0.283 | 0.099 |
| 2.5–3.0 | 9 | +0.555 | 0.103 |

The grz median difference between the `1.0–1.5` and `2.5–3.0` bins is `0.686 mag`; this contrast is within the same cluster, so a common modulus shift cannot remove it. The sign reversal is consistent with posterior contraction toward the middle of the bounded, uniform supported-luminosity prior when color gives weak luminosity information; a mismatch in the cluster color curve may also contribute. A single global median or NMAD cannot describe this model's distance quality. The paired [blind-test figure](../figures/cluster_rgb/06_ngc5466_blind_transfer_v1.png) shows the trend directly.

## Unknown abundance and physical interpretation

With the preregistered two-component UVES abundance mixture (`−1.46, −2.00 dex`), the grz median Δμ on the same 93 stars becomes `−0.155` mag at equal weights, `−0.200` mag at weights `0.8/0.2`, and `−0.094` mag at `0.2/0.8`. The median 68% interval widths are 1.44, 1.42 and 1.46 mag. Switching the conditional reference to the independently reported NGC 5466 RRc abundance `−1.89 dex` moves the grz median to `−0.100` mag. These are declared prior/abundance sensitivities. Without a measured stellar abundance, this model does not give a narrow, prior-insensitive RGB distance.

The coherent [test-cluster scale perturbations](../manifests/cluster_rgb_ngc5466_scale_sensitivity_v1.json) retain all 93 supported common sources. Changing the entire cluster's assumed `E(B−V)` by `+0.01` or `−0.01` shifts the median grz distance-modulus residual by `−0.152` or `+0.152 mag`. Coherent `+0.01 mag` changes in the g, r and z zero points shift the median by `+0.050`, `+0.040` and `−0.100 mag`, respectively; the opposite-sign tests give `−0.050`, `−0.080` and `+0.080 mag`. These are sensitivity experiments, not measured calibration errors. The external RR Lyrae modulus has its separate shared published uncertainty of `±0.09 mag`. A common cluster shift cannot erase the `0.686-mag` bright-to-faint contrast within NGC 5466.

The development-only [residual audit](../manifests/cluster_rgb_conditional_development_audit_v1.json) found grz color-residual centers near zero within M3 lower, M53 upper/lower, and NGC 7006 upper, yet the new whole-cluster test reveals large magnitude-dependent distance errors after inversion. Small forward color residuals do not establish accurate inverse distances. NGC 7006 contributes 56 gr versus 19 grz upper-RGB stars and the registered grz support criterion leaves the upper RGB untested. The same-source model and new cluster test therefore support a narrow statement: adding z helped this lower-RGB sample, while the current two-color relation and abundance prior cannot yet support a general giant-distance product.

Decision: retain this as a frozen conditional RGB diagnostic, **do not publish it as a field-star giant distance model**. The next study should first test whether extra observables supply luminosity information on the same RGB members while estimating coherent dust/zero-point uncertainties; then add an explicit RGB/HB/AGB and abundance mixture with another whole-cluster test and independent field validation. Preserve the existing cold-dwarf v4 as its separate conditional baseline.

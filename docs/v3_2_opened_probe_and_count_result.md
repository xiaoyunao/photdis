# Opened class probes and frozen numerical count audit

The v3.0 frozen probes generated probabilities for 519 candidate-admitted
native PSF objects in the opened v3.1 field cohorts. Of these, 475 have
qualified class labels. No matched identity occurs in the probe training
or temperature-calibration IDs. This comparison describes the opened
reference selection, with sky-block resampling within each role.

The strict new-sky subset includes 157 labeled objects in eight blocks,
142 of which have giant labels.

| Probe | Mean class log score, all | Mean giant probability on giants | Giant recall |
|---|---:|---:|---:|
| No color | −3.521 | 0.0272 | 0/142 |
| grz | −4.386 | 0.0280 | 2/142 |
| griz | −4.099 | 0.0343 | 2/142 |
| Multiband | −3.045 | 0.1712 | 22/142 |

For the multiband probe on giants, the paired mean giant-probability change
relative to the old target is +0.1253 (95% sky-block interval +0.0360 to
+0.1750). Relative to candidate zero it is +0.0910 (−0.0394 to +0.1605).
On all 157 labeled objects, the paired class log-score difference is
+0.7685 nat (−0.1121 to +1.3452) versus old target, +0.7267 nat
(−0.3447 to +1.4167) versus candidate zero, and −1.9821 nat
(−2.2374 to −1.4383) versus nominal reference. The frozen multiband probe
recovers some class signal lost in the target views while its probability
scores remain below nominal reference on this cohort. Compare these
results with the matched factorial and fixed-prior evidence before
choosing a mechanism.

[Object probabilities](../manifests/v3_2/opened_probes/objects.csv) ·
[Absolute scores](../manifests/v3_2/opened_probes/absolute.csv) ·
[Paired scores](../manifests/v3_2/opened_probes/paired.csv)

## Conditional count sensitivity

The allocated audit reproduced all 272 frozen count comparisons
across 34 correction states, two envelopes and four integration
variants. The 33 failed comparisons all occur for the broad
envelope under entry-level refinement and reproduce the archived
maximum mean count change.

At the worst-shift bin of each failed state, the conditional per-draw
response contribution is exactly zero. Reweighting the same 8192
draws contributes the entire signed shift. Its magnitude ranges from
0.5784 to 0.6655 object, with median 0.6117, in an envelope of 6000
objects. Weight ESS is at least 8181.75. Absolute log-normalizer
change ranges from −0.03851 to −0.03280 nat, and the span of relative
log weights is 0.2310 to 0.2778. The maximum single-object probability
change is 0.001884. These measure different aspects of convergence,
and the 33 original count violations remain in force.

[Count decompositions](../manifests/v3_2/transfer_count_audit/conditional_counts.csv) ·
[Source receipt](../manifests/v3_2/transfer_count_audit/complete.json)

[Figure and provenance](v3_2_atlas.md#opened-class-probe-and-count-audit)

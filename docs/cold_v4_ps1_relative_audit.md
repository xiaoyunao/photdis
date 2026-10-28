# PS1 relative optical check on the native final fields

## Fixed diagnostic and matching

The comparison was registered in
[`cold_v4_ps1_relative_audit.json`](../configs/cold_v4_ps1_relative_audit.json)
before reading the PS1 rows. The three native final fields and their
high-quality Gaia cold-dwarf identities had already been scored; this is
an explanatory use of them. Slurm job 60463638 ran on an allocated
Perlmutter CPU node with the frozen forest and WISE20 route. Residual
sky-block intervals and the g0/model-color decomposition were added after
the first PS1 aggregate was inspected and are exploratory diagnostics.

Gaia DR3 positions were propagated using each source's proper motion to
each candidate PS1 mean epoch. Matches had to be unique within 1 arcsec.
The PS1-to-DECam transformations from the earlier photometry audit were
applied to observed g/r/z. The fixed quality domain required PS1
`0.4<g−i<2.7`, g/r/i/z median magnitudes 15–20, at least three
measurements per band, and `0<STDEV<0.03` mag. No agreement threshold
between Legacy and PS1 was imposed, so the offsets below were not
preselected to be small.

| Field | Frozen cold anchors | PS1 1″ matches | Four-band quality | Common route acceptance after replacement |
|---|---:|---:|---:|---:|
| South RA330–340 | 5,776 | 3,638 | 1,623 | 1,613 / 1,623 (99.4%) |
| South RA20–30 | 7,097 | 7,060 | 2,734 | 2,712 / 2,734 (99.2%) |
| North RA130–140 | 2,941 | 2,923 | 1,495 | 1,485 / 1,495 (99.3%) |

The RA330 field partly lies outside the available PS1 catalog footprint:
30 of 83 queried nside32 pixels have no PS1 chunk. The quality sample is
also bluer and farther than the full frozen anchor sample. Its median
Gaia BP−RP is 1.93/1.99/1.99 and corrected parallax is
2.01/2.05/2.21 mas in the three fields. Comparisons below apply to
these PS1-qualified objects only.

## Relative optical photometry

The median observed `Legacy − transformed PS1` differences are:

| Band | South RA330 | South RA20 | North RA130 |
|---|---:|---:|---:|
| g | +0.0089 [+.0080,+.0100] | +0.0046 [+.0040,+.0051] | −0.0039 [−.0048,−.0032] |
| r | −0.0000 [−.0006,+.0005] | +0.0004 [−.0002,+.0009] | +0.0003 [−.0002,+.0010] |
| z | +0.0006 [−.0001,+.0013] | +0.0002 [−.0002,+.0006] | +0.0028 [+.0024,+.0033] |

Brackets are 95% nside64 sky-block bootstrap intervals within each field.
The g-band relative measurement/transform difference changes sign between
the southern and northern fields, while r and z stay near zero. This is a
measured relative offset in the restricted PS1 domain, not an independent
determination of which survey's absolute zero point is correct. Legacy
calibration has PS1 ancestry, and the published color transformation may
have color-dependent errors for these stars.

## Frozen-model optical substitution

For each qualified source, transformed PS1 observed g/r/z replaced
Legacy observed g/r/z. The **same catalog foreground extinction** was
subtracted, and the resulting `g0`, optical colors and `z−W1` were passed
through the unchanged color box, support calculation, WISE20 tier choice
and Legacy-trained forests. Gaia reference distance and WISE measurements
were held fixed. The experiment measures the route's sensitivity to an
optical-system change; the forest was not retrained on PS1 measurements.

| Same-source median Δμ, mag | South RA330 (n=1,613) | South RA20 (n=2,712) | North RA130 (n=1,485) |
|---|---:|---:|---:|
| Legacy input | −0.0135 [−.0261,+.0006] | −0.0480 [−.0569,−.0382] | +0.0324 [+.0217,+.0441] |
| PS1 optical input | +0.0331 [+.0238,+.0470] | −0.0144 [−.0262,−.0049] | +0.0121 [−.0008,+.0237] |
| Median of paired prediction changes | +0.0109 [+.0069,+.0171] | +0.0056 [+.0041,+.0074] | −0.0069 [−.0093,−.0047] |

The contrast between field medians contracts on this selected
subset: north minus RA330 is +0.0458 mag with Legacy and −0.0210 mag after
PS1 substitution; north minus RA20 is +0.0804 and +0.0264 mag. The
median of paired prediction changes is not the difference of the two
residual medians, because the nonlinear model can reorder stars.

At corrected parallax ≥2 mas, Legacy residual medians are
−0.0247/−0.0617/+0.0809 mag, and PS1-substituted medians are
+0.0689/−0.0122/+0.0414 mag. The PS1-qualified nearby subset has only
819/1,448/895 stars, and the substitution makes south RA330 positive.
These results show sensitivity, not a validated correction. The median
g0 shifts under substitution are −0.0089/−0.0046/+0.0040 mag; median
predicted absolute-magnitude shifts are −0.0177/−0.0048/+0.0080 mag.
The latter are larger or comparable in magnitude, so the color response
matters alongside the direct g0 term.

The [figure](../figures/cold_v4/11_ps1_relative_audit.png) displays the
optical offsets and the paired-subset residual centers. Numerical details,
including sky-block intervals, tier changes and color shifts, are in the
[aggregate manifest](../manifests/cold_v4_ps1_relative_audit_final.json).

## Decision

Relative optical photometry is now a credible contributor to the observed
field-center pattern, especially through g and the model's color
dependence. The experiment does not establish the full cause: PS1's
coverage and color cuts select a different population, its transform and
Legacy share calibration ancestry, and the forest was trained on Legacy
colors. Preserve the current model and route. The next causal test should
use an independent photometric reference with a validated cool-star
bandpass transformation, or compare same-type stars across newly
registered fields with a matched training/calibration design before any
optical correction is fitted.

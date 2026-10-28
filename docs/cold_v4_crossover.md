# Cold-v4 template and noise-field crossover

The experiment keeps the published cold-v4 forests, optical box, support
domain and `grz_clean`+WISE S/N≥20 route fixed. It reuses the exact 500 Gaia
template identities in each previously inspected RA35–45 calibration and
RA170–180 final field. For each target r bin, it pairs target noise rows from
both fields at almost identical intrinsic target `r0` (median mismatch
6×10⁻⁶–2×10⁻⁵ mag) and scores all four template×noise combinations.
The one-draw diagnostic belongs to method development; it does not replace
the frozen final test or calibrate a new interval. See
`manifests/cold_v4_crossover.json`,
`manifests/cold_v4_crossover_strata.json` and
`figures/cold_v4/06_template_noise_crossover.png`.

## Native sources and same-template degradation

| Template field | Native accepted | Native Δμ median | Native NMAD | Median source A_g | Median intrinsic r−z |
|---|---:|---:|---:|---:|---:|
| RA35–45 | 488/500 | −0.009 mag | 0.214 mag | 0.058 mag | 1.420 |
| RA170–180 | 493/500 | +0.050 mag | 0.216 mag | 0.071 mag | 1.278 |

The native median difference is 0.059 mag before either field is degraded.
The separate source-bootstrap 95% ranges are [−0.044,+0.016] and
[+0.023,+0.065] mag. The two template samples differ in color and Galactic
extinction; the native difference does not identify which underlying source
property causes it.

| Template | Noise field | Selected, r=18–19 | Δμ median / NMAD, r=18–19 | Selected, r=21–22 | Δμ median / NMAD, r=21–22 |
|---|---|---:|---:|---:|---:|
| RA35–45 | RA35–45 | 466/500 | −0.011 / 0.240 | 403/500 | +0.016 / 0.384 |
| RA35–45 | RA170–180 | 428/500 | −0.019 / 0.259 | 351/500 | +0.036 / 0.481 |
| RA170–180 | RA35–45 | 470/500 | +0.041 / 0.274 | 399/500 | +0.042 / 0.337 |
| RA170–180 | RA170–180 | 429/500 | +0.061 / 0.293 | 333/500 | +0.051 / 0.429 |

For the *same accepted template identities*, swapping RA35–45 noise for
RA170–180 noise changes the median residual by −0.000/−0.005/+0.005/−0.008
mag across r=18–22 for the first template, and
+0.001/−0.007/+0.001/−0.011 mag for the second. The paired bootstrap
95% ranges at r=21–22 are [−0.033,+0.026] and [−0.034,+0.011] mag. The
noise-field swap increases deep NMAD by 0.097 and 0.092 mag and reduces
full-route acceptance by 10.4 and 13.2 percentage points, respectively.

The deepest selected sources change model tier frequently: 334–387 of
328–398 paired sources in the four cells. The tier-change term has NMAD
0.13–0.14 mag; fixed-tier measurement noise has NMAD 0.14–0.28 mag. The
per-source identity is exactly
`degraded residual = native residual + tier-change term + fixed-tier
noise term`. Consequently, the deep scatter increase involves both routing
and photometric noise. The median of each term is not additive in general;
the identity holds source by source.

## Measured strata

- The selected used-color uncertainty exceeds 0.10 mag for only one source
  in each deepest RA170–180 noise cell and for none in the other cells.
  This sample therefore tests behavior below that error scale.
- At r=21–22, sources with native r−z≥1.4 have NMAD 0.57–0.59 mag under
  RA170–180 noise versus 0.20–0.22 mag for native r−z<0.8. This points to
  color-dependent scatter and motivates source-color-matched tests.
- The two native fields have different source A_g distributions with limited
  overlap. A dust trend measured separately inside either field cannot by
  itself establish a cross-field extinction error.

## Decision for the next experiment

The 2×2 swap localizes most of the *median* field difference to the Gaia
template populations; the noise field controls much of the deep scatter and
acceptance. Keep the published route and interval status unchanged. Match
native sources between fields using observed colors and A_g without using
distance residuals in the matching, then test an expanded set of newly
registered development/calibration/final fields if the population offset
persists. The current one-draw crossover does not support a global center
shift or a single widening factor.

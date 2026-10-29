# Latest Pro review: frozen v4 mechanism checks

Base commit: `88f7330`. The v4 forests, color box, support, WISE20 policy,
native final identities and deep-interval decisions remain frozen. This round
uses already inspected fields for diagnosis and development fields for any new
route rule. New final validation is required before replacing v4.

## 1. Common-sample crossover correction

The existing crossover arrays were reanalysed without new noise draws. The
published paired change is `median(R_1-R_0)`; the population-center comparison
is `median(R_1)-median(R_0)`. Both are now reported with joint source bootstrap
intervals, together with each environment's NMAD on the identical accepted
sources. Full-flow acceptance and NMAD remain in the manifest.

| Template | Target r | Common n | Difference of medians, mag (95% bootstrap) | Median paired difference, mag | Common NMAD environment 0 → 1, mag |
|---|---:|---:|---:|---:|---:|
| 0 | 18–19 | 406 | −0.004 [−0.032,+0.024] | −0.000 | 0.237 → 0.259 |
| 0 | 19–20 | 411 | +0.031 [−0.008,+0.061] | −0.005 | 0.274 → 0.292 |
| 0 | 20–21 | 366 | +0.024 [−0.020,+0.056] | +0.005 | 0.284 → 0.344 |
| 0 | 21–22 | 294 | +0.020 [−0.040,+0.075] | −0.008 | 0.391 → 0.460 |
| 1 | 18–19 | 409 | +0.014 [−0.003,+0.050] | +0.001 | 0.281 → 0.293 |
| 1 | 19–20 | 409 | −0.003 [−0.030,+0.023] | −0.007 | 0.300 → 0.293 |
| 1 | 20–21 | 363 | −0.021 [−0.066,+0.017] | +0.001 | 0.303 → 0.360 |
| 1 | 21–22 | 276 | −0.023 [−0.070,+0.033] | −0.011 | 0.324 → 0.429 |

The center contrast can be about 0.03 mag even where the paired-change median
is near zero. Every displayed difference-of-medians interval includes zero.
The common-sample scatter rise at the deepest bin remains visible after
holding source identity fixed. See `manifests/cold_v4_crossover_common.json`.

## 2. Weak-parallax selection correction

For the full observed-SNR <10 sample, the forest PIT remains `F(Y)/F(10σ)`.
The negative, 0–5 and 5–10 observed-SNR subgroups now use their own lower and
upper conditional CDF bounds. The center statistic's null is simulated from
the frozen weighted forest plus each star's Gaia error, conditional on the
same observed-SNR <10 cut; 1,000 simulated field realizations use identical
source identities. This null does not model other DESI/Gaia targeting cuts.

On the frozen WISE20 route, the development field has 607 weak objects and a
measured center of +0.0191 mas versus null median +0.0247 mas (null 95% range
+0.0030 to +0.0469 mas). The two populated final fields have 202 and 222
objects: measured centers +0.0402 and +0.0410 mas, null medians +0.0326 and
+0.0373 mas, and two-sided simulation p-values 0.63 and 0.81. The other final
field has no weak objects under the route. Thus the approximately +0.04 mas
statistic is ordinary under this selected null. The full-sample conditional
low-PIT fractions remain 5.4%, 5.9% and 7.2% across populated fields. See
`manifests/cold_v4_weak_parallax_v3.json`.

## 3. Next diagnostics

The fallback audit records WISE missing/invalid measurements, low S/N,
WISE quality failure and reliable high-dimensional support conflict as
different reasons. It will compare a conflict marker and a temporary hold
rule on development labels and ordinary clean candidates, including Gaia
cold anchors, DESI cold dwarfs/giants and calibrated metal-poor dwarfs.
Support-boundary stability is measured by perturbing native five-band
magnitudes with their reported errors. The rule remains a diagnostic
candidate until separate validation.

Next: force A/B/C on identical supported native-field stars, improve
cross-field matching of complete inputs, and measure fixed-tier optical
band responses on qualified PS1 counterparts.

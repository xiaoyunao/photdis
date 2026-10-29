# Latest Pro review: frozen v4 mechanism checks

Base commit: `88f7330`. The v4 forests, color box, support, WISE20 policy,
native final identities and deep-interval decisions remain frozen. This round
uses already inspected fields for diagnosis and development fields for any new
route rule. New final validation is required before replacing v4.

Extinction is already applied in every modeled band through the catalog
transmission: `m0 = 22.5 - 2.5 log10(flux) + 2.5 log10(MW_TRANSMISSION)`.
The input colors and Gaia-derived Mg reference use these corrected values.
The current low-extinction validation cut remains E(B−V)<0.03. PS1
substitutions retain the same source extinction correction.

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

## 3. Diagnostic design

The fallback audit separates WISE missing/invalid measurements, low S/N,
WISE quality failure and reliable high-dimensional support conflict. It
compares a conflict marker and a temporary hold on development labels and
ordinary clean candidates, including Gaia cold anchors, DESI cold
dwarfs/giants and calibrated metal-poor dwarfs. Native five-band magnitude
perturbations test support-boundary stability. Section 6 reports the result.

Sections 4–5 compare A/B/C on identical supported native-field stars, match
complete inputs across fields, and measure fixed-tier optical responses on
qualified PS1 counterparts.

## 4. Fixed-tier optical response

Job 60476003 completed in 3m44s. PS1 quality selects 1,623/2,734/1,495
frozen accepted stars in RA330/RA20/RA130. Each band experiment uses the
same stars and original selected model tier. Here the residual is predicted
minus Gaia distance modulus, and scatter is NMAD in mag.

| Field | Legacy center / NMAD | g-only PS1 center / NMAD | grz PS1 center / NMAD |
|---|---:|---:|---:|
| RA330 | −0.0135 / 0.2070 | +0.0383 / 0.2042 | +0.0341 / 0.1990 |
| RA20 | −0.0474 / 0.2022 | −0.0126 / 0.2174 | −0.0139 / 0.2202 |
| RA130 | +0.0322 / 0.2064 | +0.0121 / 0.2046 | +0.0120 / 0.2041 |

The g-only substitution reproduces most of the center response. r-only
centers are −0.0111/−0.0453/+0.0338 mag and z-only centers are
−0.0109/−0.0461/+0.0312 mag. Scatter and tails are retained for every band
in `manifests/cold_v4_optical_response.json`; RA20's scatter increases.
The paired grz shift medians are only +0.0119/+0.0054/−0.0069 mag, while
their NMAD is 0.066/0.060/0.055 mag. Thus shifts of population centers
cannot be reconstructed from the paired-shift median alone.

For +0.01 mag imposed in g, median fixed-tier modulus responses are
−0.0199/−0.0207/−0.0215 mag; the response NMAD is 0.044/0.046/0.047 mag.
The g term acts both directly in modulus and through shared input colors.
For the same +0.01 mag perturbation in r, the median responses are
+0.0173/+0.0209/+0.0105 mag and the response NMAD is
0.048/0.048/0.046 mag. Thus r is also a strong response channel despite
not appearing directly in the distance modulus. In the interior-support
stratum (support ratio <0.7), +0.01 mag in g changes the fixed-tier modulus
by more than 0.05 mag for 555/1,484, 987/2,547 and 523/1,377 stars
(37–39%); this is not confined to the support boundary.
Corresponding +0.01 mag z responses have median +0.0015/+0.0023/+0.0032
mag. The manifest includes both signs at 0.005/0.01/0.02 mag, color and
support-edge strata, and all fixed-tier and rerouted comparisons. These
diagnostic perturbations are not adopted photometric corrections.

## 5. Same-star A/B/C and complete-input matching

Job 60476083 completed in 1m12s. All three tiers are quality-qualified and
supported for 5,072/6,175/2,590 native stars in RA330/RA20/RA130, representing
87.8%/87.0%/88.1% of frozen accepted stars in each field.

| Field | A center / NMAD | B center / NMAD | C center / NMAD | Median paired μA−μC |
|---|---:|---:|---:|---:|
| RA330 | −0.0463 / 0.2706 | −0.0488 / 0.2477 | −0.0170 / 0.2216 | −0.0208 |
| RA20 | −0.0699 / 0.2640 | −0.0751 / 0.2291 | −0.0543 / 0.2063 | −0.0093 |
| RA130 | +0.0737 / 0.2709 | +0.0950 / 0.2546 | +0.0684 / 0.2157 | +0.0056 |

All entries are mag. The northern positive center and southern negative
centers occur in A as well as C. Adding infrared changes the response and
reduces scatter, while the field contrast already exists in optical A.

The outcome-blind match now requires identical selected tier, each actual
input color within 0.02 mag, and each used-band log error ratio within 0.5.
It leaves 1,445 and 1,177 C pairs for north versus RA330 and RA20. Their
median north-minus-south residual differences are +0.1078/+0.0634 mag;
reference Mg differences have medians +0.0930/+0.0438 mag, while model Mg
differences have medians −0.0007/−0.0044 mag. These are separate medians;
the exact additive decomposition holds pair by pair. A has 94/83 pairs,
while B has only 6/2. Parallax and brightness strata and achieved color
differences are in `manifests/cold_v4_same_star_tiers.json`.

The matched reference-label distribution remains different close to the
same complete inputs. Optical measurement/transform effects, Gaia labels,
extinction and population differences remain mechanisms to distinguish;
the experiment does not justify a constant north/south correction.

## 6. Fallback reason and type-risk experiment

Job 60476326 completed on a NERSC CPU in 24m30s. Its development fields are
RA205, RA60 and RA235; none of the native final fields was used to choose a
rule. The audit distinguishes missing/invalid WISE measurements, low S/N,
WISE quality failure, and good-quality measurements outside the next cold-
dwarf support domain. Raw WISE masks, contamination fractions and observation
counts were retrieved for all flagged objects and accepted cold giants.

| Development field | Clean v4 candidates | Infrared conflicts | Share flagged | Gaia cold anchors lost by hard hold | DESI cold dwarfs lost by hard hold |
|---|---:|---:|---:|---:|---:|
| RA205 | 71,460 | 7,808 | 10.9% | 44/1,257 (3.5%) | 149/1,116 (13.4%) |
| RA60 | 147,122 | 17,724 | 12.0% | 272/6,594 (4.1%) | 21/143 (14.7%) |
| RA235 | 363 | 50 | 13.8% | 0/9 | 1/5 |

In RA205, a hard hold would also remove 10/12 accepted DESI cold giants,
9/18 known DESI nonstars, and 4/12 calibrated metal-poor cold dwarfs. The
other development fields have no accepted cold giants, so this giant contrast
is based on 12 objects. Of the 12 RA205 giant records, 11 have high-S/N W1
that is outside B support; 10 of those actually fell back to A, while the
other retained C support. Their W1/W2 masks are zero and their WISE
contamination fractions are below one. The remaining A giant fails the WISE
quality condition. The per-object mask, contamination and support values are
in `manifests/cold_v4_fallback_development.json`.

Independent perturbations of the measured g/r/z/W1/W2 magnitudes propagate
their shared-color covariance. Of 5,000 conflicts sampled in each of RA205 and RA60,
49.9% and 48.7% respectively remain outside the relevant support domain in
at least 90% of 64 draws. Thus a hard rejection at the current radius
boundary would discard many cold dwarfs and include many noise-sensitive
border cases. The selected next method is to **retain the frozen v4
distance and add an infrared-support-conflict/type-risk flag**. A hard hold
and any revised support threshold need separate development and independent
validation. The flag does not assign a giant probability.

The v4 box still misses the previously identified bluer metal-poor dwarfs.
That extension remains a separate branch requiring its own training coverage.

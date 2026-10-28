# Frozen cold-v4 native transfer: registered independent fields

## Design and frozen inputs

The model, optical color box, A/B/C feature support, strict Gaia cold-anchor
definition and `grz_clean` + WISE S/N ≥20 C→B→A policy were fixed before
this round. `configs/cold_v4_native_transfer_fields.json` registered two
southern final fields; `configs/cold_v4_native_transfer_north_final.json`
registered one northern final field after the two informative development
fields and **before inspecting the northern source rows**. Its explicit
hypothesis was a more positive native distance-modulus residual in the
northern field than in both southern final fields, especially for corrected
parallax ≥2 mas. The two originally registered final fields were scored
once after the same parallax-stratum sky-block uncertainty code was fixed.
The low-yield RA235–245 development field (nine selected stars) and the
RA180–190 field with zero non-reserved cold anchors remain in the manifests.

All comparisons below use `Δμ = μ̂ − μ_Gaia`, with positive values placing a
star too far away. Only strictly selected, high-quality Gaia cold anchors
enter these native-brightness metrics. Uncertainty is a 400-replicate
nside=64 sky-block bootstrap within each field; it measures within-field
sampling variation, not the full between-field variation.

| Field and role | Strict anchors | Selected cold anchors | Median Δμ [sky-block 95%], mag | NMAD, mag |
|---|---:|---:|---:|---:|
| RA60–70, Dec −35 to −25, south development | 9,431 | 6,594 | −0.0059 [−0.0139,+0.0023] | 0.2056 |
| RA230–240, Dec +25 to +35, north development | 7,128 | 4,554 | +0.0415 [+0.0335,+0.0509] | 0.2146 |
| RA330–340, Dec −35 to −25, south final | 7,829 | 5,776 | −0.0178 [−0.0266,−0.0059] | 0.2310 |
| RA20–30, Dec −25 to −15, south final | 9,095 | 7,097 | −0.0520 [−0.0590,−0.0454] | 0.2186 |
| RA130–140, Dec +25 to +35, north final | 4,105 | 2,941 | +0.0687 [+0.0535,+0.0812] | 0.2261 |

The registered directional hypothesis is satisfied in this set of fields:
the northern final median exceeds the two southern final medians by
0.0865 and 0.1207 mag. This is a **field contrast**, not an established
hemisphere-wide correction or a causal latitude effect. All three final
fields have roughly 96% acceptance among their strict cold anchors; about
91–92% of their accepted sources use tier C.

## Parallax, color, magnitude and extinction diagnostics

| Final field | 1–2 mas: n, median [95%], mag | ≥2 mas: n, median [95%], mag |
|---|---|---|
| South RA330 | 1,876, −0.0078 [−0.0204,+0.0052] | 3,899, −0.0221 [−0.0357,−0.0087] |
| South RA20 | 2,204, −0.0223 [−0.0317,−0.0123] | 4,891, −0.0662 [−0.0774,−0.0570] |
| North RA130 | 800, −0.0174 [−0.0268,−0.0102] | 2,141, +0.1121 [+0.0990,+0.1270] |

The largest contrast occurs in the nearer ≥2 mas stratum. In the 1–2 mas
stratum, all three final medians are close to zero and negative. The northern
final median also rises with source `r0`: +0.009, +0.111 and +0.214 mag in
the <17, 17–18 and ≥18 bins, respectively. The RA20 southern field is
negative in all three bins (−0.026, −0.082, −0.049 mag). This pattern merits
a source-population and astrometric-label check before any correction.

In Gaia BP−RP bins 1.8–2.3, 2.3–2.8 and 2.8–3.5, the northern medians are
+0.079, +0.111 and +0.134 mag. Corresponding RA20 medians are −0.049,
−0.096 and −0.106 mag. Thus the contrast persists among red cold stars,
although Gaia color is a diagnostic here and is not a model input.

Median corrected parallax is 2.42, 2.46 and 2.71 mas in the two southern
and northern final fields; median Gaia BP−RP is 2.21, 2.24 and 2.24. Median
source `A_g` is 0.058, 0.050 and 0.080 mag. Extinction is already corrected
using each source's DR11 `MW_TRANSMISSION` in all model input bands and the
`g0` distance-modulus term (see `docs/extinction_audit.md`). Within the
southern RA20 final field, the low/high `A_g=0.05` groups remain negative
(−0.062/−0.041 mag); the northern field has only 36 low-`A_g` sources.
These marginal strata do not isolate a dust cause.

### Outcome-blind matched-source diagnostic

On the three final fields *after* their one-time scoring, we matched
individual northern and southern selected stars without using residuals
in the matching. Greedy unique pairs used their eight nearest candidates
and standardized Euclidean radius ≤2. Pair and northern-sky-block
bootstraps were kept separate. These comparisons are explanatory use of
already inspected final fields and cannot validate a newly fitted method.

| North minus south, after matching | South RA330 | South RA20 |
|---|---:|---:|
| Legacy `g-r`, `r-z`, source `A_g`: pairs, median Δresidual [north sky-block 95%] | 1,962, +0.055 [+0.036,+0.070] | 1,361, +0.078 [+0.058,+0.101] |
| Add corrected parallax, Gaia BP−RP, source `r0`: pairs, median [95%] | 2,016, +0.046 [+0.033,+0.062] | 1,470, +0.076 [+0.060,+0.092] |
| Same six-feature match, ≥2 mas only: pairs, median [95%] | 1,453, +0.081 [+0.059,+0.095] | 1,068, +0.104 [+0.087,+0.119] |

For the all-parallax six-feature pairs, median absolute differences are
0.017 in `g-r`, 0.014 in `r-z`, 0.0022 mag in `A_g`, ~0.10 mas in corrected
parallax, ~0.020 mag in Gaia BP−RP and ~0.08 mag in `r0`. The contrast
persists under this measured-property control, especially for ≥2 mas.
The bootstrap groups northern sky positions; shared southern blocks and
matching uncertainty are not fully represented. See
`manifests/cold_v4_native_transfer_final_match.json` for all matching
variants, including the 1–2 mas subgroup.

### Astrometric quality and spectroscopy

The selected stars in south RA330, south RA20 and north RA130 have median
Gaia RUWE 1.004/1.016/1.013, formal parallax error
0.0815/0.0786/0.0872 mas, and corrected-parallax S/N
27.7/29.5/28.3. The *applied* Gaia zero-point terms have medians
−0.0345/−0.0336/−0.0370 mas. Their ~0.003 mas difference corresponds
to only a few millimagnitudes at typical 2–3 mas parallax, far below the
observed ~0.09–0.12 mag field contrast; this comparison does not bound
unmodeled spatial Gaia systematics.

Restricting to RUWE≤1.1 leaves 5,476/6,723/2,761 selected sources with
residual medians −0.009/−0.046/+0.073 mag. In the ≥2 mas and RUWE≤1.1
subset, the medians are −0.016/−0.060/+0.120 mag. The field contrast is
therefore present among the better Gaia single-source fits.

The DESI quality, unambiguous cold-dwarf, valid metallicity-calibration
subset contains only 0/13/35 selected stars in these fields. The latter
two have similar calibrated median [Fe/H] (−0.319/−0.303 dex), but they
are tiny, selected spectroscopic samples. Their median residuals
(−0.030/−0.035 mag) differ from their full-field medians, so they do not
measure the metallicity distribution of the photometric anchors. See
`manifests/cold_v4_native_transfer_final_labels.json`.

The figure `figures/cold_v4/08_native_transfer_final.png` plots the
development and final medians. Development ≥2 mas points have no error bars
because their per-stratum sky bootstrap was added after their original run;
final points use the fixed per-stratum bootstrap.

## Decision

The final fields independently reproduce a spatially varying native
distance-modulus center. The previous 2×2 experiment attributes most deep
scatter and acceptance loss to noise/tier changes, but a native field-center
difference remains. Matching the old RA35/RA170 templates on optical colors,
extinction, corrected parallax, Gaia BP−RP and Gaia G did not remove their
center contrast. Neither result identifies whether the new contrast arises
from population/metallicity, photometric calibration, extinction-map error,
Gaia astrometric systematics, or a combination. No global center shift,
new interval calibration, forest refit or new stellar branch is justified
by these field summaries alone. Retain the frozen high-quality cold-dwarf
conditional distance and its field-dependent systematic limitation; do not
publish a nominal deep interval from the old calibration.

Next diagnostic: map the center across the six additional disjoint fields
registered in `configs/cold_v4_native_transfer_sky_census.json` before
source inspection. This is a diagnostic sky census under the same frozen
route, not an independent test of a new correction. Any subsequent
correction requires new registered calibration and final sky regions.

## Provenance

- Frozen policy SHA256: `a569489a9055ad2acc3aa6e6b623b03e846bd266bd15aea8f8a66dc6d4cfdb1d`.
- Southern final registration SHA256: `2a32f23eb8bc0ee6ebe1999819acf3be929c15e67d51f99d44b3445bee5f3b75`.
- Northern final registration SHA256: `f1760873af3256a1e38d2a70317f1493b86b4c6b34165ca421e6e9c90a53eb79`.
- Numerical evidence: `manifests/cold_v4_native_transfer_final.json` and
  `manifests/cold_v4_native_transfer_north_final.json`; matched-source
  evidence: `manifests/cold_v4_native_transfer_final_match.json`;
  astrometric and DESI strata: `manifests/cold_v4_native_transfer_final_labels.json`.
- All field scans and scorings ran on allocated Perlmutter compute nodes;
  source catalogs and trained forest objects remain on NERSC.

# Frozen native-distance sky census

## Registered diagnostic design

Six disjoint 10°×10° fields were fixed in
`configs/cold_v4_native_transfer_sky_census.json` before their source rows
were inspected. The cold-v4 forests, optical box, support domains,
strict-anchor definition and `grz_clean`+WISE S/N≥20 route were unchanged.
This is a spatial diagnostic following the one-time final-field test in
`docs/cold_v4_native_transfer_final.md`. It does not fit a center
correction. `Δμ=μ̂−μ_Gaia`; positive means the model places a star farther
away than the Gaia reference. CIs resample nside64 sky blocks within fields.

| RA, Dec ranges (deg) | Center Galactic b | Strict Gaia anchors | Selected cold anchors | Median Δμ [sky-block 95%], mag | NMAD, mag | ≥2 mas: n, median Δμ |
|---|---:|---:|---:|---:|---:|---:|
| 10–20, −35 to −25 | −86.6° | 6,926 | 5,656 | −0.0092 [−0.0149,−0.0009] | 0.2108 | 4,094, −0.0143 |
| 310–320, −35 to −25 | −39.6° | 0 | 0 | — | — | 0, — |
| 340–350, −15 to −5 | −58.7° | 324 | 216 | +0.0068 [−0.0238,+0.0413] | 0.2001 | 155, +0.0196 |
| 140–150, +25 to +35 | +48.2° | 7,588 | 5,331 | +0.0390 [+0.0335,+0.0469] | 0.2209 | 3,843, +0.0663 |
| 200–210, +25 to +35 | +79.0° | 7,707 | 5,395 | +0.0496 [+0.0403,+0.0583] | 0.2168 | 3,528, +0.0756 |
| 250–260, +25 to +35 | +36.0° | 4,288 | 2,218 | +0.0775 [+0.0656,+0.0893] | 0.2696 | 1,166, +0.1709 |

The RA310 field had 4,907,889 PSF rows at |b|>30°, and **all failed**
the unchanged `E(B−V)<0.03` cut. RA340 had 3,992,271 PSF rows at
|b|>30°, of which 106,379 passed that dust cut; its 216 selected anchors
therefore have a wide field-median interval. Both fields remain in the
registered result.

The three positive-b fields have positive overall and ≥2 mas medians.
RA10, the informative negative-b census field, is near zero and slightly
negative. This extends the observed spatial center variation, with
RA250 giving the largest nearby-anchor offset and NMAD. The median does
not rise monotonically with Galactic latitude: RA250 (b≈+36°) is more
positive than RA200 (b≈+79°). In the 1–2 mas stratum, all three northern
medians are −0.018, −0.005 and +0.005 mag, again much closer to zero than
their ≥2 mas medians. See
`figures/cold_v4/09_native_sky_census.png` and
`manifests/cold_v4_native_transfer_sky_census.json`.

The six centers also have matching signs of Galactic and ecliptic
latitude. Before their scores were read, four further disjoint fields
with *opposite* coordinate signs were fixed in
`configs/cold_v4_native_transfer_cross_sign.json`: two b>0/β<0 and two
b<0/β>0. Their registered comparison will check which sky coordinate
sign, if either, better organizes the native residuals. All ten fields
remain diagnostics under the same frozen method.

## Reproducibility

- Six-field construction: NERSC Slurm arrays 60460898 and 60461294,
  all six tasks completed on allocated CPUs. The initial regular-queue
  array 60460329 and its dependent score were cancelled while pending.
- Frozen scoring: NERSC job 60461354, completed on allocated CPU.
- Full per-source tables and models remain on NERSC; aggregate manifests,
  registration, code, report and selected figure are synchronized.

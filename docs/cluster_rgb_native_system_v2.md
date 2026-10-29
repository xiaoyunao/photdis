# Native-system and band-matched RGB audit

## Scope and data definition

This audit follows the Pro review of `beb15b7`. The pooled quadratic A/B
models, their once-only NGC 7006 test, and the cold-dwarf models remain at
their frozen identities. The new measurements use the four existing cluster
membership parents at `Pmember≥0.5`, matched by exact Gaia source ID to DR11
sweeps on a Slurm compute node. The core sequence uses `Pmember>0.9` and the
Gaia-CMD stage screens registered in
[`cluster_rgb_phase_regions_v1.json`](../configs/cluster_rgb_phase_regions_v1.json).
These stage labels remain broad RGB candidates.

## Native passbands and extinction

DR11 southern `RELEASE=11010` is DECam in `g,r,z`; northern `11011` is
BASS/90Prime in `g,r` and MzLS/Mosaic3 in `z`. The sweep files contain
`RELEASE` but no per-row `PHOTSYS`. The [DR11 description](https://www.legacysurvey.org/dr11/description/)
states that measured fluxes remain in each instrument's natural AB system;
the [release mapping](https://www.legacysurvey.org/release/) gives the exact
camera assignment. Thus M3, M53 and NGC 7006 share the southern optical
passbands, while NGC 2419 uses northern passbands. They are displayed with
separate labels and are not pooled into a new fitted curve.

The same DR11 description publishes PS1-to-camera calibration polynomials
valid for standard stars with `0.4<g-i<2.7`. Evaluated over that range,
their **northern minus southern predicted `g-r`** differs by about
`0.020–0.047 mag` for the *same PS1 colors*. This is a passband sensitivity
scale, not a correction applied to the cluster giants. Some RGB stars are
outside the published color range, and a camera conversion would require
its own giant-SED validation. A `0.02 mag` color change on a relation with
`|dM_g/d(g-r)|≈7.7` changes the inferred magnitude by about `0.15 mag`.

The project already corrects extinction: `m₀=22.5−2.5 log10(FLUX)
+2.5 log10(MW_TRANSMISSION)`. The DR11 catalog describes
`MW_TRANSMISSION_*` as its Galactic transmission based on SFD `E(B−V)`;
the existing cluster audit measures `A_g/E(B−V)=3.214`, `A_r/E(B−V)=2.165`,
`A_z/E(B−V)=1.211` in all three development systems. Therefore
`M_g=g₀−μ_cluster`, `(g−r)₀=g₀−r₀`, and `(r−z)₀=r₀−z₀` use dereddened
Legacy photometry. The external distance modulus and dust-scale error
shift all stars of one cluster together.

## Band-matched selection and measurement

The new `gr_clean` tier uses positive finite flux, inverse variance,
MW transmission, at least one contributing exposure and `S/N≥5` in
**each of g and r**. It also applies the existing per-band limits
`FRACMASKED<0.4`, `FRACIN>0.7`, `FRACFLUX<0.5`, the g/r `SATUR` and
`ALLMASK` bits, and the global `BAILOUT` bit. It does not use z.
For comparison the old `grz_clean` definition is unchanged, and its
original selected row is retained separately from the new gr-preferred row.
The new rows and full per-source diagnostics stay on NERSC; the linked
manifest below records small aggregate results.

The plotted `g-r` sequence uses the newly gr-qualified A sample. The
`r-z` sequence uses the exact frozen grz-qualified B sample. M3 contributes
its lower RGB, M53 contributes its upper and lower RGB, and NGC 2419
contributes its upper RGB. M3 upper RGB and NGC 2419 lower/RHB-overlap
candidates are kept as separate diagnostic layers. Horizontal bin bars
show the 16–84% **within-bin color range**; they include curve slope,
measurement noise and stage overlap and are not an intrinsic RGB width.

For a local color-minus-curve residual `e=c−f(M_g)`, the propagated optical
variance is `Var(e)=Var(c)+f′² σ_g²−2f′ Cov(c,M_g)`.
For `c=g-r`, `Cov(c,M_g)=σ_g²`; for `c=r-z`, it is zero under independent
band errors. The two colors also share `Cov(g-r,r-z)=−σ_r²`.
The measured local residual width is compared with these propagated
errors and with radius and image-quality strata. Whole-cluster modulus and
dust uncertainties are reported separately as correlated scale shifts.

## Results

The allocated sweep rejoin and cached-table finalization completed on
Slurm jobs `60496210` and `60496669`. The first job's FITS byte-string
stage labels caused an empty-locus serialization error after all four
member tables had been written; this was corrected, tested and finalized
from those tables without rescanning the source sweeps. All selected gr
rows have the same `LS_ID` as the frozen grz row for the matched source.
Release counts confirm 25,395/9,826/640 southern members in M3/M53/
NGC 7006 and 1,563 northern members in NGC 2419, with the expected
`11010`/`11011` assignments.

| Cluster and Gaia-CMD stage | Members `P>0.9` | New clean gr | Frozen clean grz | Added because z fails on same row | gr `M_g` 5–95% |
|---|---:|---:|---:|---:|---:|
| M3 upper RGB | 387 | **97** | 2 | 95 | +0.42 to +0.86 |
| M3 lower RGB | 579 | 480 | 422 | 58 | +0.97 to +2.75 |
| M53 upper RGB | 357 | 246 | 156 | 90 | −1.01 to +0.64 |
| M53 lower RGB | 313 | 259 | 253 | 6 | +0.82 to +2.80 |
| NGC 2419 upper RGB | 458 | 405 | 391 | 14 | −1.88 to +0.49 |
| NGC 2419 lower/RHB overlap | 93 | 81 | 78 | 3 | −0.16 to +1.38 |
| NGC 7006 upper RGB, diagnostic only | 84 | 56 | 19 | 37 | −1.35 to +0.47 |

Thus z alone removes many otherwise gr-qualified stars, especially in
M3 and M53. For M3, changing the input-band rule recovers 95 upper-region
candidates, **but the bright RGB tip remains absent**: the recovered
stars are concentrated near the faint edge of the Gaia-defined upper
region. The other 290 upper candidates still fail a g or r measurement,
fraction or mask condition. This does not justify loosening the g/r
quality rule without an additional measurement check.

The new [stage-aligned locus figure](../figures/cluster_rgb/04_stage_matched_native_loci_v2.png)
uses 480 M3 lower and 505 M53 upper+lower gr candidates, plus 405 NGC 2419
upper candidates. Its `r-z` panel uses the unchanged frozen B cores of
422, 409 and 391. The 81 new NGC 2419 lower-region stars remain a separate
red-HB-overlap diagnostic and are not in the plotted upper sequence.

At seven common `M_g` grid points from `+1.0` to `+2.5`, the two southern
**lower-RGB** gr sequences differ by median
`(g-r)₀(M3)−(g-r)₀(M53)=+0.0275 mag` (grid range about +0.019 to +0.034).
The six common grz `r-z` grid points give `+0.0232 mag`. These are
interpolated same-luminosity differences, not passband effects, since
both systems use DECam. Perturbing the whole M3/M53 modulus by their
quoted combined standard errors shifts the gr comparison by about
`±0.0026/±0.0025 mag`. The registered illustrative dust shifts of
`±0.005/±0.010` in `E(B−V)` move it by about `±0.0063/±0.0125 mag`.
These common shifts must not be divided by the number of member stars.

The local gr residual NMAD around each system's own binned curve is
`0.016` (M3 lower), `0.015` (M53 upper+lower) and `0.090 mag`
(NGC 2419 upper), versus median formal propagated optical errors of
`0.0012`, `0.0016` and `0.0050 mag`. These descriptive residuals are
computed against curves estimated from the same stars and are **not** an
intrinsic scatter estimate. The NGC 2419 gr residual center differs with
radius: the inner half (`r<1.52′`) has median `−0.078 mag`, and the outer
half `+0.043 mag`; inner/outer NMADs are `0.083/0.049 mag`. The higher
residual width and spatial trend point to a combination of phase overlap,
crowding or image measurement effects that cannot be assigned uniquely
from this audit. The `r-z` local residual NMAD is also `0.091 mag` there.

An independent *imaging-measurement* check used the published DR11
PS1-to-BASS/MzLS polynomials on the northern upper-RGB sample. Of 391
strict grz targets, 193 have a unique PS1 neighbor within 1″ and 25 pass
the preregistered four-band magnitude, observation, scatter and `g-i`
range checks. For those 25, Legacy minus transformed PS1 has median
`Δg=−0.034`, `Δr=−0.011`, `Δz=−0.003 mag`, and `Δ(g-r)=−0.025 mag`.
The inner/outer halves have median `Δ(g-r)=−0.035/−0.010 mag` at a
`2.84′` radius split. This small quality subset supports checking
crowded-field photometry before attributing NGC 2419 width to stellar
metallicity. PS1 helped calibrate Legacy zero points, so this is not an
independent absolute zero-point determination.

The full aggregate measurements are in
[`cluster_rgb_system_audit_v2.json`](../manifests/cluster_rgb_system_audit_v2.json)
and the [NGC 2419 PS1 check](../manifests/cluster_rgb_ngc2419_ps1_check_v1.json).
The large per-star FITS tables remain only on NERSC. The analysis uses only
already registered clusters; NGC 7006 is an audit system here, having
already served as the frozen A/B whole-system test.

## Next model design

A photo-only giant model would represent
`p(color | M_g, [Fe/H], RGB stage, camera system)` and marginalize over
unknown abundance and phase for ordinary targets. Before fitting it,
collect at least one further northern BASS/MzLS cluster spanning upper and
lower RGB at metallicity comparable with a southern cluster, plus a
southern DECam upper/lower system at a different measured metallicity.
The sample design needs overlapping `M_g` ranges for each system and phase,
an external cluster modulus, a dust-scale uncertainty, and member identity
and image-quality evidence. Register at least one **new** untouched
whole-cluster test before fitting; a separate test in each camera system
would directly check transfer. NGC 7006 has already been examined.
The current clean W1+W2 counts of 9, 22 and 10 across the three
development systems are only acceptance diagnostics.
